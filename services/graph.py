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

from services.tools import buscar_catalogo_tool, enviar_notificacion_tool, consultar_fidelidad_tool
from core.phone_utils import is_meta_identifier, get_meta_directive_prompt
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
    "REGLA OBLIGATORIA DE MULTIMEDIA:\n"
    "Cuando envíes una imagen o video (tipo 'imagen' o 'video'), el campo 'mensaje' es OBLIGATORIO y debe contener SIEMPRE un texto descriptivo o caption relevante. "
    "NUNCA omitas el campo 'mensaje' ni lo dejes como cadena vacía (\"\") en imágenes o videos. "
    "Si no tienes un caption natural, escribe algo breve como '¡Aquí te comparto la carta!' o una descripción del contenido.\n\n"
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

    # Construir tools habilitadas dinámicamente y reglas condicionales
    available_tools = []
    reglas_herramientas_extra = ""
    
    if tools_cfg.get("buscar_catalogo", True):
        available_tools.append(buscar_catalogo_tool)
        reglas_herramientas_extra += (
            "⚠️ REGLA ESTRICTA E INQUEBRANTABLE (RAG):\n"
            "¡TIENES QUE USAR TU HERRAMIENTA DE CATÁLOGO (`buscar_catalogo_tool`) EN EL 100% DE LOS PEDIDOS!\n"
            "Sin importar si ya crees saber el precio, DEBES usar la herramienta SIEMPRE para confirmar precios actuales, tamaños y disponibilidad ANTES de dar el detalle del pedido. ¡NUNCA INVENTES PRECIOS!\n\n"
        )
        
    if tools_cfg.get("enviar_notificacion", True):
        available_tools.append(enviar_notificacion_tool)
        reglas_herramientas_extra += (
            "⚠️ REGLA OBLIGATORIA DE NOTIFICACIÓN DE PEDIDO (`enviar_notificacion_tool`):\n"
            "1. CUÁNDO INVOCAR:\n"
            "- Cuando el cliente confirme o cierre su pedido (productos elegidos, total, método de pago y dirección de entrega acordados).\n"
            "- Si el cliente MODIFICA o AGREGA productos al pedido en curso, o aclara/cambia su método de pago (ej. efectivo, con cuánto paga para el cambio, comprobante de Nequi), o precisa su dirección/celular, DEBES invocar la herramienta para que el sistema actualice el pedido en el dashboard y notifique al equipo.\n"
            "- Si el cliente indica expresamente que desea hacer un pedido completamente NUEVO e INDEPENDIENTE para otra persona o dirección distinta, pon `es_pedido_nuevo_separado=True`.\n"
            "2. CUÁNDO NO INVOCAR:\n"
            "- Si el pedido ya fue confirmado y notificado y el cliente solo responde con un agradecimiento ('gracias', 'listo', 'ok') sin cambiar productos, método de pago ni dirección, NO vuelvas a invocar la herramienta.\n"
            "3. FORMATO OBLIGATORIO:\n"
            "- En `lista_productos` escribe UNA SOLA LÍNEA concisa por cada producto (`[Cantidad]x [Nombre del producto] - $[Subtotal]`), "
            "SIN repetir líneas separadas de 'Precio Unidad', 'Subtotal' ni 'Total sin el valor del domicilio'.\n"
            "4. VINCULACIÓN DE FIDELIDAD:\n"
            "- Pasa siempre `cliente_id` y `cliente_cedula` si fueron obtenidos con `consultar_fidelidad_tool`.\n"
            "- Si se redimió beneficio, pasa `beneficio_fidelidad='DESCUENTO_15'` o `'GRANIZADO_GRATIS'`.\n\n"
        )

    if tools_cfg.get("consultar_fidelidad", True):
        available_tools.append(consultar_fidelidad_tool)
        reglas_herramientas_extra += (
            "⚠️ REGLAS ESTRICTAS DE TARJETA DE FIDELIDAD (`consultar_fidelidad_tool`):\n"
            "1. CONSULTA SILENCIOSA:\n"
            "- Al inicio de la conversación de pedido, consulta SILENCIOSAMENTE usando el número de WhatsApp del cliente (`consultar_fidelidad_tool(identificador='')`).\n"
            "- Si existe: salúdalo con su nombre (ej: '¡Hola [Nombre]!') y coméntale sus sellos actuales (ej: 'Llevas [X]/10 sellos en tu Tarjeta de Fidelidad Q\\'hubo Mor 🍧').\n"
            "2. SI NO ESTÁ REGISTRADO POR TELÉFONO O NO TIENE CÉDULA/TARJETA DE IDENTIDAD:\n"
            "- Pregúntale amablemente si tiene Tarjeta de Fidelidad de Q\'hubo Mor o si desea suministrarnos su número de cédula o tarjeta de identidad para buscarlo.\n"
            "- Si responde que no tiene cédula, no tiene tarjeta de identidad, no la recuerda o no está registrado, INVÍTALO ACTIVAMENTE A INSCRIBIRSE GRATIS EN LA WEB explicándole que es súper fácil, gratis y toma solo 30 segundos, brindándole los enlaces oficiales:\n"
            "  * Registro y Tarjeta: https://qhubomor.com/fidelidad\n"
            "  * Misiones con sellos gratis: https://qhubomor.com/misiones\n"
            "- ¡REGLA DE ORO: NUNCA BLOQUEES EL PEDIDO! Si no desea registrarse en ese momento o no da documento, continúa tomando su domicilio con total normalidad.\n"
            "3. LÍMITE DE SELLOS Y PAUSAS (CHECKPOINTS EN 5 Y 10):\n"
            "- El contador se pausa en 5 hasta que el cliente disfrute su descuento del 15% (no sube a 6 en esa misma compra).\n"
            "- El contador se pausa en 10 hasta que el cliente reclame su Granizado 12oz Gratis (no se reinicia a 1 o 2 en esa compra).\n"
            "4. CÁLCULO Y COTIZACIÓN DEL 15% DE DESCUENTO:\n"
            "- El 15% de descuento aplica ÚNICAMENTE a Granizados. JAMÁS a comidas, bebidas en botella ni al valor del domicilio.\n"
            "- Redondeo al millar hacia abajo: `descuento = (int(subtotal_granizados * 0.15) // 1000) * 1000` (Ejemplo: $3.750 -> $3.000; $4.500 -> $4.000).\n"
            "- Cotiza el total con el descuento ya aplicado y aclara: 'La cajera validará tu descuento de fidelidad al despachar'.\n"
            "5. GRANIZADO GRATIS (HITO 10):\n"
            "- Si el cliente tiene 10 sellos, pregúntale: '¿Deseas tu Granizado 12oz GRATIS con licor o sin licor? ¿Y qué sabor te gustaría?'.\n"
            "- Anota en la lista: '1x Granizado 12oz [Sabor] (Con/Sin Licor) - $0 [PREMIO FIDELIDAD 10/10]'.\n\n"
        )
        
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

    # ── Inyección completa del RAG en el system prompt (sin búsqueda por similitud) ──
    # Se carga TODO el catálogo del bot de una sola vez y se inyecta en el prompt.
    # Ventajas vs similarity_search:
    #   - Sin latencia de embeddings ni consultas vectoriales
    #   - Todos los productos siempre visibles, sin riesgo de que falte alguno
    #   - Ideal para menús/catálogos pequeños-medianos (típico de un restaurante)
    api_key = bot_config.get("api_key", "")
    rag_context_block = ""
    if tools_cfg.get("buscar_catalogo", True) and api_key:
        try:
            from services.rag_service import get_all_knowledge
            bot_id_for_rag = bot_config.get("bot_id", 0)
            all_docs = get_all_knowledge(bot_id=bot_id_for_rag, api_key=api_key)

            if all_docs:
                # Unir todos los fragmentos del catálogo en un solo texto
                catalog_text = "\n\n---\n\n".join(
                    doc["content"].strip() for doc in all_docs if doc.get("content", "").strip()
                )
                rag_context_block = (
                    "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "📋 CATÁLOGO COMPLETO DEL NEGOCIO (cargado completo — todos los productos y precios):\n"
                    "INSTRUCCIÓN: Antes de calcular el precio de cualquier producto, búscalo en este catálogo.\n"
                    "Lee el campo CATEGORÍA de cada uno para NO confundir comidas con granizados.\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"{catalog_text}\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "REGLA: Estos son los precios REALES. Úsalos para TODOS los productos del pedido, "
                    "sin importar cuántos pida el cliente a la vez."
                )
                print(f"{log_prefix} [RAG-COMPLETO] ✅ Catálogo cargado: {len(all_docs)} fragmentos ({len(catalog_text)} chars).")
            else:
                print(f"{log_prefix} [RAG-COMPLETO] ℹ️ Sin documentos RAG para este bot.")
        except Exception as rag_err:
            print(f"{log_prefix} [RAG-COMPLETO] ⚠️ No se pudo cargar el catálogo: {rag_err}")

    # Detectar si el remitente es un identificador de Meta / ManyChat / Username
    current_phone = str(bot_config.get("user_phone") or state.get("phone", "")).strip()
    meta_prompt_block = ""
    if is_meta_identifier(current_phone):
        meta_prompt_block = get_meta_directive_prompt(current_phone)
        print(f"{log_prefix} ⚠️ [META-ID DETECTADO] '{current_phone}' es un identificador de Meta/ManyChat/Username. Inyectando directiva para exigir número celular real.")
    elif current_phone and current_phone != "simulador":
        meta_prompt_block = (
            f"\n\n📌 DATO VERIFICADO DEL SISTEMA (lineaWA): El número real de WhatsApp de este cliente es '{current_phone}'. "
            f"Si el cliente escribe otro número por error de digitación en el chat, en `enviar_notificacion_tool` y en tus registros internos "
            f"prevalece SIEMPRE el número real de la línea de WhatsApp ('{current_phone}').\n"
        )

    # Lógica para inyectar System Prompt como primer mensaje si no existe al inicio
    if not messages or not isinstance(messages[0], SystemMessage):
        final_system_prompt = system_prompt + "\n\n" + reglas_herramientas_extra + INSTRUCCIONES_JSON + rag_context_block + meta_prompt_block
        messages = [
            SystemMessage(content=final_system_prompt)
        ] + messages
    elif meta_prompt_block and meta_prompt_block not in str(messages[0].content):
        # Si ya existe un SystemMessage previo, asegurar que la directiva de Meta quede inyectada
        messages[0] = SystemMessage(content=messages[0].content + meta_prompt_block)

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
_all_tools = [buscar_catalogo_tool, enviar_notificacion_tool, consultar_fidelidad_tool]
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
