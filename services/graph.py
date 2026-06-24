"""
services/graph.py
─────────────────────────────────────────────────────────────
Fuente de verdad del historial: los JSON de main.py (sin cambios).
LangGraph recibe el historial convertido como mensajes iniciales en
cada llamada. De esta forma:
  - Los JSON siguen siendo la fuente única de verdad.
  - LangGraph tiene siempre el contexto completo de la conversación.
  - No hay dos verdades divergentes.
"""
import json
from typing import TypedDict, Annotated, List, Dict, Any, Optional

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from services.tools import buscar_catalogo_tool, enviar_notificacion_tool
try:
    from google.genai import types as genai_types
except ImportError:
    genai_types = None

# Máximo de mensajes del historial JSON que se inyectan como contexto.
# Evita exceder la ventana de contexto en conversaciones muy largas.
MAX_HISTORY_MESSAGES = 30


# ── Estado del grafo ──────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    bot_id: int
    system_prompt: str
    phone: str
    bot_config: Dict[str, Any]


# ── Respuesta estructurada ────────────────────────────────────────────────────

# ── Nodo del agente ───────────────────────────────────────────────────────────

INSTRUCCIONES_JSON = (
    "REGLAS DE BÚSQUEDA Y HERRAMIENTAS:\n"
    "Si el usuario pregunta por información en tiempo real, eventos recientes, el clima, noticias, "
    "horarios o cualquier dato que requiera buscar en internet, DEBES usar la herramienta de búsqueda de Google ANTES de generar tu respuesta final.\n"
    "Mientras uses herramientas, NO respondas en formato JSON todavía. Simplemente invoca la herramienta necesaria.\n\n"
    "RESPUESTA FINAL AL USUARIO:\n"
    "Solo cuando hayas obtenido la información de las herramientas (o si no fue necesario usarlas), "
    "debes dar tu respuesta FINAL ÚNICAMENTE en formato JSON válido de acuerdo a la estructura que se te ha indicado anteriormente.\n"
    "Asegúrate de que la salida sea ÚNICAMENTE el JSON, sin bloques de código markdown ni texto adicional."
)


def _build_llm(api_key: str, ai_model: str, temperature: float = 0.5, thinking_budget: int = 0, thinking_level: str = "HIGH"):
    """Construye el LLM con la api_key y modelo del cliente."""
    
    kwargs = {
        "model": ai_model,
        "temperature": temperature,
        "google_api_key": api_key,
    }
    
    # Manejo de razonamiento (thinking) directo desde la request
    # Google ha modificado el SDK: los modelos 2.0 (incluso 2.0-flash-thinking) usan budget numérico,
    # mientras que los modelos 3.0+ usan levels de texto (MINIMAL, LOW, MEDIUM, HIGH)
    if "gemini-3" in ai_model:
        # Usamos directamente el thinking_level tal como viene ("MINIMAL", "LOW", "MEDIUM", "HIGH")
        kwargs["model_kwargs"] = {"thinking_config": {"thinking_level": thinking_level}}
    else:
        # Los modelos 2.x (incluyendo 2.0-flash-thinking) usan thinking_budget numérico
        # Se envía siempre a menos que sea 0 (desactivado)
        if thinking_budget != 0:
            kwargs["model_kwargs"] = {"thinking_budget": thinking_budget}

    return ChatGoogleGenerativeAI(**kwargs)


def call_model(state: AgentState):
    messages = state["messages"]
    system_prompt = state.get("system_prompt", "Eres un asistente de IA.")
    bot_config = state.get("bot_config", {})

    api_key = bot_config.get("api_key", "")
    ai_model = bot_config.get("ai_model", "gemini-2.5-flash")
    tools_cfg = bot_config.get("tools_config", {})
    thinking_budget = bot_config.get("thinking_budget", 0)
    thinking_level = bot_config.get("thinking_level", "HIGH")

    # Construir tools habilitadas dinámicamente
    available_tools = []
    if tools_cfg.get("buscar_catalogo", True):
        available_tools.append(buscar_catalogo_tool)
    if tools_cfg.get("enviar_notificacion", True):
        available_tools.append(enviar_notificacion_tool)
        
    # Añadir herramientas nativas de Google (Search, Maps) si están habilitadas
    use_google_search = bot_config.get("use_google_search", False)
    use_google_maps = bot_config.get("use_google_maps", False)
    
    log_prefix = f"[{bot_config.get('userbot', 'bot')}/{bot_config.get('phone', 'phone')}]"

    if use_google_search or use_google_maps:
        if genai_types is not None:
            if use_google_search:
                print(f"{log_prefix} [TOOLS] Habilitando Búsqueda de Google en LangGraph")
                available_tools.append({"google_search": {}})
                if use_google_maps:
                    print(f"{log_prefix} [TOOLS-WARN] Se solicitaron Google Search y Google Maps. La API no permite combinarlos. Se priorizará Google Search.")
            elif use_google_maps:
                # Nota: google_maps
                print(f"{log_prefix} [TOOLS] Habilitando Google Maps en LangGraph")
                available_tools.append({"google_maps": {}})
        else:
            print(f"[{bot_config.get('userbot')}/{bot_config.get('phone')}] WARNING: google.genai is not installed, cannot use Google Search/Maps.")

    # Lógica para inyectar System Prompt como primer mensaje si no existe al inicio
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [
            SystemMessage(content=system_prompt + "\n\n" + INSTRUCCIONES_JSON)
        ] + messages

    # Obtener modelos de fallback si vienen de config_dict, o construir lista local
    fallback_models = bot_config.get("fallback_models", [])
    models_to_try = []
    if ai_model and ai_model.strip():
        models_to_try.append(ai_model.strip())
    
    seen = set(models_to_try)
    for fm in fallback_models:
        if fm not in seen:
            models_to_try.append(fm)
            seen.add(fm)
            
    if not models_to_try:
        models_to_try = ["gemini-2.5-flash"]
        

    llm_with_tools = None
    for idx, current_model in enumerate(models_to_try):
        try:
            print(f"{log_prefix} [LANGGRAPH {idx + 1}/{len(models_to_try)}] Intentando con modelo: {current_model}")
            llm = _build_llm(api_key, current_model, temperature=0.5, thinking_budget=thinking_budget, thinking_level=thinking_level)
            
            # Validar si estamos usando Google Search/Maps y usar la nueva sintaxis
            has_native_google_tools = False
            if genai_types is not None:
                has_native_google_tools = any(
                    isinstance(t, dict) and ("google_search" in t or "google_maps" in t)
                    for t in available_tools
                )
            
            if has_native_google_tools:
                # Gemini 3.5 requires this config when mixing custom tools and native built-in tools
                gemini_tool_config = {
                    "function_calling_config": {"mode": "AUTO"},
                    "include_server_side_tool_invocations": True
                }
                llm_with_tools = llm.bind_tools(available_tools, **{"tool_config": gemini_tool_config})
            else:
                llm_with_tools = llm.bind_tools(available_tools) if available_tools else llm
            
            response = llm_with_tools.invoke(messages)
            
            print(f"{log_prefix} [RAW OUTPUT FROM SDK - MODEL: {current_model}]:")
            print(response.content)

            # Debug para ver si se activaron herramientas
            if hasattr(response, 'tool_calls') and response.tool_calls:
                print(f"{log_prefix} [TOOLS-ACTIVATION] 🔧 El modelo decidió activar las siguientes herramientas: {[t.get('name') for t in response.tool_calls]}")
                for t in response.tool_calls:
                    print(f"{log_prefix}    -> Tool: {t.get('name')} | Args: {t.get('args', {})}")
            else:
                print(f"{log_prefix} [TOOLS-ACTIVATION] ⏸️ El modelo NO activó ninguna herramienta en este turno.")

            # Check for grounding metadata in response attributes
            if hasattr(response, 'response_metadata') and response.response_metadata:
                grounding_metadata = response.response_metadata.get('groundingMetadata') or response.response_metadata.get('grounding_metadata')
                if grounding_metadata:
                    print(f"{log_prefix} [TOOLS-SUCCESS] Metadata de Grounding detectada! El modelo buscó en Google u obtuvo datos externos.")
                else:
                    if has_native_google_tools:
                        print(f"{log_prefix} [TOOLS-INFO] Las herramientas estaban habilitadas, pero el modelo NO hizo uso de ellas para esta consulta.")

            print(f"{log_prefix} ✅ ÉXITO con modelo {current_model} en LangGraph.")
            
            return {"messages": [response]}
        except Exception as e:
            print(f"{log_prefix} ⚠️ Fallo en LangGraph con modelo {current_model}: {e}")
            continue

    raise RuntimeError("All fallback models failed in LangGraph")

# ── Construcción del grafo (sin checkpointer: stateless por diseño) ───────────

def _build_graph(tools_list: list):
    tool_node = ToolNode(tools_list)

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    # SIN checkpointer: el historial viene de los JSON, no de SQLite
    return workflow.compile()


# Grafo compilado (genérico; tools se aplican dinámicamente en call_model)
_all_tools = [buscar_catalogo_tool, enviar_notificacion_tool]
app_graph = _build_graph(_all_tools)


# ── Conversión de historial JSON → mensajes LangChain ────────────────────────

def _json_history_to_lc_messages(
    history: List[Dict[str, Any]],
    max_messages: int = MAX_HISTORY_MESSAGES
) -> List[BaseMessage]:
    """
    Convierte los últimos `max_messages` entradas del historial JSON
    al formato LangChain que entiende LangGraph.

    Roles del JSON:
      "cliente" / "usuario_pausado"  →  HumanMessage
      "asistente" / "operador"       →  AIMessage
    """
    human_roles = {"cliente", "usuario_pausado"}
    ai_roles = {"asistente", "operador"}

    # Tomar los últimos N para no exceder la ventana de contexto
    recent = history[-max_messages:] if len(history) > max_messages else history

    lc_messages: List[BaseMessage] = []
    for entry in recent:
        role = entry.get("role", "")
        content = entry.get("mensaje", "")
        if not content:
            continue
            
        # Filtrar entradas basura generadas por errores de parseo (Bug #1 y #2)
        if "Respuesta IA no textual" in content:
            continue
            
        if role in human_roles:
            lc_messages.append(HumanMessage(content=f"[CLIENTE]: {content}"))
        elif role in ai_roles:
            lc_messages.append(AIMessage(content=f"[ASISTENTE]: {content}"))

    return lc_messages


# ── Punto de entrada principal ────────────────────────────────────────────────

def process_message_with_graph(
    bot_id: int,
    phone: str,
    message: str,
    system_prompt: str,
    config_dict: Dict[str, Any],
    json_history: Optional[List[Dict[str, Any]]] = None,
    user_push_name: str = None,
    fecha_hora_formateada: str = None
) -> str:
    """
    Punto de entrada principal para procesar un mensaje con LangGraph.
    
    - params:
        bot_id - ID del bot para usarlo en tool configs si es necesario
        phone - Teléfono del usuario
        message - Mensaje actual del usuario
        system_prompt - Prompt del sistema extraído de la BD/configuración
        config_dict - Diccionario con TODA la configuración del bot (para sacar keys, cx, config de tools, etc.)
        json_history - Historial de la BD o memoria del webhook
        user_push_name - Nombre (pushName) del usuario en WhatsApp.
        fecha_hora_formateada - String con la fecha y hora local del usuario.
    """
    # 1. Convertir historial JSON a mensajes LangChain
    history_messages = _json_history_to_lc_messages(json_history or [])

    print(f"[{bot_id}/{phone}] [HISTORIAL] Entradas JSON recibidas: {len(json_history or [])}")
    print(f"[{bot_id}/{phone}] [HISTORIAL] Mensajes convertidos a LangChain: {len(history_messages)}")
    if history_messages:
        print(f"[{bot_id}/{phone}] [HISTORIAL] Primer msg (antiguo): {str(history_messages[0].content)[:100]}...")
        print(f"[{bot_id}/{phone}] [HISTORIAL] Último msg (reciente): {str(history_messages[-1].content)[:100]}...")

    # 2. Agregar el mensaje actual al final, incluyendo el pushName
    if not fecha_hora_formateada:
        fecha_hora_formateada = datetime.now().strftime("%d/%m/%Y %I:%M %p")
    user_name_part = f" ({user_push_name})" if user_push_name else ""
    formatted_user_content = f"""Mensaje ACTUAL del cliente{user_name_part} (puede incluir texto de varios mensajes cortos concatenados, y/o la descripción de un archivo multimedia procesado): {message} -> [Fecha y hora actual - {fecha_hora_formateada}]"""

    all_messages = history_messages + [HumanMessage(content=formatted_user_content)]

    # 3. Estado inicial
    input_state: AgentState = {
        "messages": all_messages,
        "bot_id": bot_id,
        "system_prompt": system_prompt,
        "phone": phone,
        "bot_config": config_dict,
    }

    # 4. Ejecutar el grafo (sin thread_id porque no hay checkpointer)
    last_message = None
    config = {
        "configurable": {
            "bot_id": bot_id,
            "phone": phone,
            "config_dict": config_dict
        }
    }
    import time
    start_time = time.time()
    print(f"[{bot_id}/{phone}] [TIMING] Iniciando app_graph.stream() en LangGraph...", flush=True)
    
    for event in app_graph.stream(input_state, stream_mode="values", config=config):
        last_message = event["messages"][-1]

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"[{bot_id}/{phone}] [TIMING] ⏱️ app_graph.stream() terminó en {elapsed:.2f} segundos.", flush=True)

    if not last_message:
        return "{}"
        
    # Extraer metadata de grounding_metadata si usó Google Search
    try:
        rm = last_message.response_metadata
        if "grounding_metadata" in rm or (hasattr(last_message, "additional_kwargs") and "grounding_metadata" in last_message.additional_kwargs):
            print(f"[{bot_id}/{phone}] 🔍 INFO: El modelo usó Google Search para esta respuesta.")
    except Exception:
        pass

    content = last_message.content
    if isinstance(content, list):
        text_blocks = [blk["text"] for blk in content if isinstance(blk, dict) and "text" in blk]
        return "".join(text_blocks)
    
    return str(content)
