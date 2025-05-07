import asyncio
import json
import os
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import pytz
import babel.dates
# No necesitamos 'locale' importado directamente para babel.dates con locale=idioma
# import locale
import time # Importar time para trabajar con timestamps en segundos/milisegundos

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ----- Configuration -----
BASE_DIR = "./Download/AutoSystem"
HIST_BASE_DIR = os.path.join(BASE_DIR, "historial") # Directorio base para historiales
CONF_DIR = os.path.join(BASE_DIR, "conf_2") # Directorio para configuraciones (asumo que es necesario)
PAUSED_STATUS_DIR = os.path.join(BASE_DIR, "paused_status") # Nuevo directorio para el estado de pausa por contacto

# Crear directorios base si no existen
for d in [BASE_DIR, HIST_BASE_DIR, CONF_DIR, PAUSED_STATUS_DIR]:
    os.makedirs(d, exist_ok=True)

# Basic mapping from country name (or common variants) to a pytz timezone string
COUNTRY_TIMEZONE_MAP = {
    "argentina": "America/Argentina/Buenos_Aires",
    "bolivia": "America/La_Paz",
    "brasil": "America/Sao_Paulo", # Example, need to be more specific for Brazil
    "chile": "America/Santiago",
    "colombia": "America/Bogota",
    "costa rica": "America/Costa_Rica",
    "cuba": "America/Havana",
    "ecuador": "America/Guayaquil",
    "el salvador": "America/El_Salvador",
    "guatemala": "America/Guatemala",
    "honduras": "America/Tegucigalpa",
    "mexico": "America/Mexico_City",
    "nicaragua": "America/Managua",
    "panama": "America/Panama",
    "paraguay": "America/Asuncion",
    "peru": "America/Lima",
    "puerto rico": "America/Puerto_Rico",
    "republica dominicana": "America/Santo_Domingo",
    "españa": "Europe/Madrid",
    "uruguay": "America/Montevideo",
    "venezuela": "America/Caracas",
    "usa": "America/New_York", # Example, needs to be more specific for USA
    "canada": "America/Toronto", # Example, needs to be more specific for Canada
    "uk": "Europe/London",
    "france": "Europe/Paris",
    "germany": "Europe/Berlin",
    "italy": "Europe/Rome",
    "portugal": "Europe/Lisbon",
    "australia": "Australia/Sydney", # Example, needs to be more specific for Australia
    "india": "Asia/Kolkata",
    "china": "Asia/Shanghai",
    "japan": "Asia/Tokyo",
    # Add more countries as needed
}

# Helper function to get timezone string (defined here for completeness)
def get_timezone_from_country(country_name: str) -> str:
    """Maps a country name to a pytz timezone string."""
    # Convert country name to lower case for consistent lookup
    country_name_lower = country_name.lower()
    # Use the predefined map, with a fallback to UTC if not found
    return COUNTRY_TIMEZONE_MAP.get(country_name_lower, "UTC")


# ----- Models -----
class MessageRequest(BaseModel):
    lineaWA: str
    mensaje_reciente: str
    userbot: str # userbot es necesario para organizar historiales
    apikey: str # Gemini API key
    server: str # WhatsApp server URL (e.g., http://100.42.185.2:8012). Used for sending messages.
    numerodemensajes: int
    promt: str
    token: str # WhatsApp server token. Used for sending messages and potentially reaction check.
    pais: str = Field(..., description="Country name (e.g., 'Colombia', 'Mexico', 'España')")
    idioma: str = Field(..., description="Language code (e.g., 'es' for Spanish, 'en' for English)")
    delay_seconds: float = Field(7.0, description="Time in seconds to wait before processing grouped messages")
    # Nuevos campos para la configuración de generación de la IA
    temperature: float = Field(0.5, description="Controls randomness. Lowering results in less random completions. Range: 0.0 to 1.0.")
    topP: float = Field(0.95, description="Nucleus sampling. Controls the diversity of outputs by sampling from the most probable tokens.")
    maxOutputTokens: int = Field(4096, description="The maximum number of tokens to generate in the response.")
    # Nuevo campo para el modo pausa automático
    pause_timeout_minutes: int = Field(0, description="Automatic pause duration in minutes based on reactions. 0 means no automatic timeout.")
    # Nuevo campo para especificar el modelo de IA
    ai_model: str = Field("gemini-2.0-flash", description="The AI model to use for generating responses (e.g., 'gemini-2.0-flash', 'gemini-1.0-pro').")
    # Assuming isGroup will always be False for this pause logic based on the prompt
    # isGroup: bool = Field(False, description="Indicates if the chat is a group chat.") # Add if group chat pause is needed later


class DeleteHistoryRequest(BaseModel):
    userbot: str = Field(..., description="The userbot whose history needs to be deleted.")
    delete_all: bool = Field(..., description="If true, delete all history for the specified userbot. If false, delete specific lineaWAs.")
    lineaWAs_to_delete: Optional[List[str]] = Field(None, description="List of lineaWAs to delete history for, required if delete_all is false.")

# ----- Application State -----
app = FastAPI(title="WhatsApp Message Processor", version="1.0.0")

# Diccionario para rastrear tareas de procesamiento pendientes por userbot+lineaWA
# Clave: f"{userbot}_{lineaWA}" (str)
# Valor: Dict{'task': asyncio.Task, 'fragments': List[str], 'original_request': MessageRequest}
processing_tasks: Dict[str, Dict[str, Any]] = {}

# Cache para el estado de pausa de los contactos (puede ayudar al rendimiento, pero la fuente de verdad es el archivo)
# Clave: userbot (str)
# Valor: Dict[phone_number (str), state_dict (Dict)]
contact_pause_state_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}


# ----- Helper Functions for Pause State Management (File-based) -----

def get_contact_state_file_path(userbot: str, phone: str) -> str:
    """Returns the path to the JSON file for a contact's pause state."""
    userbot_paused_status_dir = os.path.join(PAUSED_STATUS_DIR, userbot)
    os.makedirs(userbot_paused_status_dir, exist_ok=True)
    return os.path.join(userbot_paused_status_dir, f"{phone}.json")

def load_contact_state(userbot: str, phone: str) -> Dict[str, Any]:
    """Loads a contact's pause state from their JSON file."""
    filepath = get_contact_state_file_path(userbot, phone)
    # Ensure userbot's cache dict exists
    if userbot not in contact_pause_state_cache:
         contact_pause_state_cache[userbot] = {}

    # Check cache first
    if phone in contact_pause_state_cache[userbot]:
        # print(f"[{userbot}/{phone}] Estado cargado desde cache.")
        return contact_pause_state_cache[userbot][phone]

    # Load from file if not in cache
    if not os.path.exists(filepath):
        # print(f"[{userbot}/{phone}] Archivo de estado no encontrado. Usando estado inicial.")
        # Initial state assumes unpaused and no relevant reaction processed yet
        initial_state = {"is_paused": False, "pause_start_time": None, "last_control_reaction_timestamp": 0}
        contact_pause_state_cache[userbot][phone] = initial_state # Cache initial state
        return initial_state
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
            # Validate structure and provide defaults if needed
            if "is_paused" not in state or not isinstance(state["is_paused"], bool):
                 state["is_paused"] = False
            if "pause_start_time" not in state:
                 state["pause_start_time"] = None
            # Ensure pause_start_time is string or None
            if state["pause_start_time"] is not None and not isinstance(state["pause_start_time"], str):
                 try:
                      # Attempt to convert if it's a datetime object somehow got saved
                      state["pause_start_time"] = state["pause_start_time"].isoformat()
                 except:
                      state["pause_start_time"] = None # Fallback

            # Use last_control_reaction_timestamp to track which reaction set the state
            if "last_control_reaction_timestamp" not in state or not isinstance(state["last_control_reaction_timestamp"], (int, float)):
                 state["last_control_reaction_timestamp"] = 0 # Use 0 for no relevant reaction timestamp


            # Cache the loaded state
            contact_pause_state_cache[userbot][phone] = state
            # print(f"[{userbot}/{phone}] Estado cargado desde archivo.")
            return state
    except (json.JSONDecodeError, FileNotFoundError, Exception) as e:
        print(f"[{userbot}/{phone}] Error loading contact state from {filepath}: {e}")
        # On error, return initial state and do NOT cache the error state
        initial_state_on_error = {"is_paused": False, "pause_start_time": None, "last_control_reaction_timestamp": 0}
        # Do NOT cache the error state to force reload from file next time or retry saving.
        return initial_state_on_error


def save_contact_state(userbot: str, phone: str, state: Dict[str, Any]):
    """Saves a contact's pause state to their JSON file."""
    filepath = get_contact_state_file_path(userbot, phone)
    try:
        # Update cache first
        if userbot not in contact_pause_state_cache:
             contact_pause_state_cache[userbot] = {}
        contact_pause_state_cache[userbot][phone] = state

        # Ensure pause_start_time is serializable
        if state.get("pause_start_time") is not None and not isinstance(state["pause_start_time"], str):
             try:
                 state["pause_start_time"] = state["pause_start_time"].isoformat()
             except:
                 state["pause_start_time"] = None # Cannot serialize, set to None

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=4)
        # print(f"[{userbot}/{phone}] Estado guardado en archivo.")
    except Exception as e:
        print(f"[{userbot}/{phone}] Error saving contact state to {filepath}: {e}")
        # Optionally, remove from cache on save error to force reload next time
        if userbot in contact_pause_state_cache and phone in contact_pause_state_cache[userbot]:
             del contact_pause_state_cache[userbot][phone]


async def get_latest_control_reaction_timestamps(userbot: str, token: str, phone: str) -> tuple[int, int]:
    """
    Fetches recent messages and returns the timestamps of the latest ✋ and ✅ reactions
    from the BOT to its OWN message. Returns (0, 0) if none are found.
    """
    log_prefix = f"[{userbot}/{phone}]"
    jid = f"{phone}@s.whatsapp.net"
    reaction_api_base_url = "http://100.42.185.2:8001"
    reaction_check_url = f"{reaction_api_base_url}/chats/{jid}?id={userbot}&cursor_fromMe=false&isGroup=false&limit=50" # Increased limit

    headers = {"x-access-token": token}

    print(f"{log_prefix} Obteniendo timestamps de últimas reacciones de control (✋, ✅)...")

    latest_pause_reaction_ts_ms = 0
    latest_unpause_reaction_ts_ms = 0

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(reaction_check_url, headers=headers, timeout=15.0)
            response.raise_for_status()
            response_data = response.json()

            if isinstance(response_data, dict) and 'data' in response_data and isinstance(response_data['data'], list):
                messages = response_data['data']

                if not messages:
                    print(f"{log_prefix} API de reacciones retornó una lista de mensajes vacía.")
                    return 0, 0

                # Iterate through messages to find the timestamps of the latest ✋ and ✅ from BOT to BOT message
                for msg in messages:
                    try:
                        if 'message' in msg and 'reactionMessage' in msg['message']:
                             reaction_data = msg['message']['reactionMessage']
                             reaction_text = reaction_data.get('text')
                             reaction_ts_ms_raw = reaction_data.get('senderTimestampMs')

                             reaction_ts_ms = 0
                             # Robust timestamp parsing
                             if isinstance(reaction_ts_ms_raw, str) and reaction_ts_ms_raw.isdigit():
                                 reaction_ts_ms = int(reaction_ts_ms_raw)
                             elif isinstance(reaction_ts_ms_raw, dict) and 'low' in reaction_ts_ms_raw and 'high' in reaction_ts_ms_raw:
                                 try:
                                     low = reaction_ts_ms_raw['low']
                                     high = reaction_ts_ms_raw['high']
                                     # Combine parts into a 64-bit integer (handle sign based on high bit)
                                     combined_ts = (high << 32) | (low & 0xFFFFFFFF)
                                     if combined_ts & (1 << 63): # If highest bit is set, it's negative in two's complement
                                         # Convert from unsigned to signed 64-bit representation if necessary
                                          if combined_ts >= (1 << 63): # Only if it's in the negative range
                                               combined_ts -= (1 << 64)

                                     reaction_ts_ms = combined_ts
                                 except Exception as ts_e:
                                      print(f"{log_prefix} Error parsing complex timestamp {reaction_ts_ms_raw}: {ts_e}")
                                      reaction_ts_ms = 0
                             # else: unexpected format, reaction_ts_ms remains 0


                             if reaction_text is None or reaction_text == '' or reaction_ts_ms <= 0:
                                  continue # Skip reactions with empty text or invalid/zero timestamp

                             # Check if the message containing the reaction is from the BOT
                             is_message_from_bot_containing_reaction = msg.get('key', {}).get('fromMe', False)

                             # Check if the parent message (the one being reacted to) was from the BOT
                             parent_msg_key = reaction_data.get('key', {})
                             is_parent_from_bot = parent_msg_key.get('fromMe', False)

                             # Only consider this reaction if it's from the BOT AND to a BOT message
                             if is_message_from_bot_containing_reaction and is_parent_from_bot:
                                  # Found a relevant reaction from BOT to BOT message
                                  if reaction_text == '✋':
                                      if reaction_ts_ms > latest_pause_reaction_ts_ms:
                                          latest_pause_reaction_ts_ms = reaction_ts_ms
                                          # print(f"{log_prefix}  > Nueva última reacción de PAUSA (✋) encontrada: {reaction_ts_ms}")
                                  elif reaction_text == '✅':
                                      if reaction_ts_ms > latest_unpause_reaction_ts_ms:
                                          latest_unpause_reaction_ts_ms = reaction_ts_ms
                                          # print(f"{log_prefix}  > Nueva última reacción de REANUDAR (✅) encontrada: {reaction_ts_ms}")


                    except Exception as e:
                        print(f"{log_prefix} Error inesperado al procesar mensaje para reacción: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

                # print(f"{log_prefix} Última reacción ✋ timestamp MS: {latest_pause_reaction_ts_ms}")
                # print(f"{log_prefix} Última reacción ✅ timestamp MS: {latest_unpause_reaction_ts_ms}")
                return latest_pause_reaction_ts_ms, latest_unpause_reaction_ts_ms


            else:
                print(f"{log_prefix} Advertencia: La respuesta del API de reacciones no tiene la estructura esperada (Dict con 'data' list). Respuesta: {response_data}")
                return 0, 0 # Return 0,0 on unexpected structure


    except httpx.RequestError as e:
        print(f"{log_prefix} Error de red llamando a reaction endpoint: {e}")
        return 0, 0
    except httpx.HTTPStatusError as e:
        print(f"{log_prefix} Error HTTP {e.response.status_code} llamando a reaction endpoint: {e.response.text}")
        return 0, 0
    except json.JSONDecodeError:
        print(f"{log_prefix} Error al decodificar la respuesta JSON del reaction endpoint.")
        return 0, 0
    except Exception as e:
        print(f"{log_prefix} Error inesperado llamando a reaction endpoint: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0


def is_contact_paused(state: Dict[str, Any], pause_timeout_minutes: int) -> bool:
    """
    Checks if a contact is currently paused based on their state dictionary and timeout.
    This function does NOT modify the state.
    """
    is_paused_flag = state.get("is_paused", False)
    pause_start_time_str = state.get("pause_start_time")
    # print(f"[{state.get('userbot', 'Unknown')}/{state.get('phone', 'Unknown')}] is_contact_paused: is_paused_flag={is_paused_flag}, pause_start_time_str={pause_start_time_str}, timeout={pause_timeout_minutes}min")


    if is_paused_flag and pause_start_time_str:
        if pause_timeout_minutes > 0:
            try:
                pause_start_time = datetime.fromisoformat(pause_start_time_str)
                timeout_duration = timedelta(minutes=pause_timeout_minutes)
                if datetime.now() - pause_start_time > timeout_duration:
                    # print(f"[{state.get('userbot', 'Unknown')}/{state.get('phone', 'Unknown')}] Pausa automática expirada.")
                    return False # Timeout expired, effectively unpaused
                else:
                     # print(f"[{state.get('userbot', 'Unknown')}/{state.get('phone', 'Unknown')}] Contacto está PAUSADO (timeout activo).")
                     return True # Still paused, timeout not expired
            except ValueError:
                 print(f"[{state.get('userbot', 'Unknown')}/{state.get('phone', 'Unknown')}] Error: pause_start_time en estado inválido '{pause_start_time_str}'. Tratando como no pausado.")
                 return False # Invalid pause time, treat as not paused
            except Exception as e:
                 print(f"[{state.get('userbot', 'Unknown')}/{state.get('phone', 'Unknown')}] Error inesperado en is_contact_paused: {e}. Tratando como no pausado.")
                 return False
        else:
            # print(f"[{state.get('userbot', 'Unknown')}/{state.get('phone', 'Unknown')}] Contacto está PAUSADO (sin timeout automático).")
            return True # Paused, no timeout set
    else:
        # print(f"[{state.get('userbot', 'Unknown')}/{state.get('phone', 'Unknown')}] Contacto no está pausado.")
        return False # Not paused according to the flag


# Definición de hist_file_path (Asegurarse de que esté definida globalmente o accesible)
def hist_file_path(userbot: str, phone: str) -> str:
    """Devuelve la ruta completa al archivo de historial para un userbot y número de teléfono."""
    userbot_hist_dir = os.path.join(HIST_BASE_DIR, userbot)
    # Asegurarse de que el directorio del userbot exista
    os.makedirs(userbot_hist_dir, exist_ok=True)
    return os.path.join(userbot_hist_dir, f"{phone}.txt")


# ----- Function for Media Endpoint Call -----
async def call_media_endpoint(server: str, session: str, token: str, gemini_api_key: str, telefono: str) -> Optional[Dict[str, Any]]:
    """
    Calls the external media processing endpoint.
    """
    # Use the provided media endpoint URL
    media_api_url = "http://100.42.185.2:8011/get-last-media/"
    headers = {"Content-Type": "application/json"} # Assume JSON request
    # The token from the main WA server might be needed by the media endpoint
    # Add token to headers if the media endpoint expects it there, otherwise include in body
    # headers["x-access-token"] = token

    log_prefix = f"[{session}/{telefono}]"
    print(f"{log_prefix} Llamando al endpoint de procesamiento de medios en {media_api_url}...")

    # Construct the payload based on the expected parameters - CORRECTION: Use "sesion" instead of "session"
    payload = {
        "sesion": session, # Corrected key name
        "token": token, # Include token in body as it's available in MessageRequest
        "gemini_api_key": gemini_api_key, # Include API key as it might be needed for AI processing on media side
        "telefono": telefono
    }

    try:
        async with httpx.AsyncClient() as client:
            # Use a reasonable timeout for potentially longer media processing
            response = await client.post(media_api_url, json=payload, timeout=60.0)

        response.raise_for_status() # Raise for 4xx/5xx errors

        response_data = response.json()
        print(f"{log_prefix} Respuesta del media endpoint: {response_data}")

        # Validate basic structure of the response - expecting a dictionary
        if isinstance(response_data, dict):
             # Return the dictionary as is, process_message_final will handle its content and success status
             return response_data
        else:
             # If the response is not a dictionary, consider it an invalid response
             print(f"{log_prefix} Advertencia: Respuesta inesperada del media endpoint (no es un diccionario).")
             return {"procesamiento_exitoso": False, "error_detail": "Respuesta inválida del media endpoint"}


    except httpx.RequestError as e:
        # Handle network errors (connection refused, timeout, etc.)
        print(f"{log_prefix} Error de red al llamar al media endpoint: {e}")
        return {"procesamiento_exitoso": False, "error_detail": f"Error de red al llamar al media endpoint: {e}"}
    except httpx.HTTPStatusError as e:
        # Handle HTTP errors (400, 404, 500, etc.)
        print(f"{log_prefix} Error HTTP {e.response.status_code} al llamar al media endpoint: {e.response.text}")
        return {"procesamiento_exitoso": False, "error_detail": f"Error HTTP {e.response.status_code} al llamar al media endpoint: {e.response.text}"}
    except json.JSONDecodeError:
        # Handle errors during JSON decoding
        print(f"{log_prefix} Error al decodificar la respuesta JSON del media endpoint.")
        return {"procesamiento_exitoso": False, "error_detail": "Respuesta no JSON válida del media endpoint"}
    except Exception as e:
        # Handle any other unexpected errors
        print(f"{log_prefix} Error inesperado al llamar al media endpoint: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for unexpected errors
        return {"procesamiento_exitoso": False, "error_detail": f"Error inesperado al llamar al media endpoint: {e}"}

# ----- End New Function -----


async def process_message_final(req: MessageRequest, full_message: str, pais: str, idioma: str, ai_model: str) -> Optional[Dict[str, Any]]:
    """
    Procesa el mensaje combinado final: llama a la IA, actualiza el historial.
    Incorpora timezone y locale, y configuración de generación dinámica.
    Implementa múltiples estrategias de reintento para la llamada a la IA.
    Utiliza el modelo de IA especificado.
    Devuelve la respuesta de la IA (éxito o fallback) o None si hay un fallo irrecuperable ANTES de llamar a la IA.
    """
    phone = req.lineaWA
    userbot = req.userbot
    log_prefix = f"[{userbot}/{phone}]"
    print(f"{log_prefix} Iniciando procesamiento final (país: {pais}, idioma: {idioma}, modelo IA: {ai_model}) con mensaje: '{full_message}'")

    # 1. Cargar historial
    # hist_file_path is defined globally or in an accessible scope now
    history_path = hist_file_path(userbot, phone)
    hist = {}
    try:
        if os.path.exists(history_path):
            with open(history_path, 'r', encoding='utf-8') as f:
                hist = json.load(f)
    except json.JSONDecodeError:
        print(f"{log_prefix} Advertencia: El archivo de historial {history_path} no es un JSON válido. Se creará uno nuevo.")
    except Exception as e:
        print(f"{log_prefix} Error al cargar el historial {history_path}: {e}")


    # 2. Construir texto del historial para el prompt
    plain_history = ""
    if hist:
        try:
            client_keys = sorted([int(k.split('_')[-1]) for k in hist if k.startswith('cliente_m_') and k.split('_')[-1].isdigit()])
        except ValueError:
             client_keys = []
             print(f"{log_prefix} Advertencia: Claves de historial inesperadas encontradas, ignorando para orden.")


        last_indices_to_show = client_keys[-req.numerodemensajes:] if req.numerodemensajes > 0 else client_keys

        for i in sorted(last_indices_to_show):
            cli_msg = hist.get(f"cliente_m_{i}", '')
            res_msg = hist.get(f"respuesta_{i}", '')

            plain_history += "=============================\n"
            plain_history += f"🗣️ Cliente (cliente_m_{i}):\n{cli_msg}\n\n"
            if res_msg:
                plain_history += f"🤖 Respuesta (respuesta_{i}):\n"
                plain_history += f"    {res_msg}\n"
            else:
                plain_history += "🤖 Respuesta: (sin respuesta registrada)\n"
            plain_history += "=============================\n\n"


    # 3. Verificar si se necesita llamar al endpoint de medios y procesar la respuesta
    # The full_message passed here already contains the combined text fragments.
    # We now check for media indicators *in this combined message*.
    final_content_for_ai = full_message

    # Emojis a detectar para cada tipo de medio según el prompt
    media_emoji_map = {
        '🎤': 'audio',
        '📷': 'imagen',
        '📄': 'documento',
        '📹': 'video'
    }

    detected_media_type = None
    for emoji, media_type in media_emoji_map.items():
        if emoji in full_message:
            detected_media_type = media_type
            print(f"{log_prefix} Mensaje combinado contiene indicador de medio: {emoji} ({media_type}). Llamando a media endpoint.")
            break # Only check for the first detected emoji

    # Call the call_media_endpoint function here if media is detected
    if detected_media_type:
        # The call_media_endpoint function is now defined above
        media_response = await call_media_endpoint(
            server=req.server, # Use the server from the request (assumed main WA server)
            session=req.userbot,
            token=req.token,
            gemini_api_key=req.apikey,
            telefono=phone
        )

        if media_response and media_response.get("procesamiento_exitoso"):
            media_type_from_response = media_response.get("tipo")
            processed_text = media_response.get("procesamiento_gemini")
            media_url = media_response.get("url")
            file_name = media_response.get("nombre_archivo_local")

            media_description_for_ai = f"\n[Información de medio adjunto (Tipo: {media_type_from_response}):"

            if processed_text:
                # For audio and likely documents, the processed text is the key information
                media_description_for_ai += f" Texto procesado: '{processed_text.strip()}'"
            elif media_url:
                # For image/video, the URL might be relevant, or just indicating its presence
                media_description_for_ai += f" URL: {media_url}"
            elif file_name:
                 # For documents, the file name is also relevant
                 media_description_for_ai += f" Nombre de archivo: {file_name}"
            else:
                 media_description_for_ai += " No se obtuvo texto procesado o URL/nombre."

            media_description_for_ai += "]"

            # Append the media description to the content for the AI
            final_content_for_ai += media_description_for_ai
            print(f"{log_prefix} Información de medio añadida al contenido para la IA: {media_description_for_ai}")

        elif media_response and not media_response.get("procesamiento_exitoso"):
             error_detail = media_response.get("error_detail", "Unknown error")
             print(f"{log_prefix} Media endpoint respondió pero el procesamiento no fue exitoso. Detalle: {error_detail}")
             final_content_for_ai += f"\n[Información de medio adjunto (Tipo: {detected_media_type}): Error al procesar el medio: {error_detail}]"
        else:
            # This case is handled by the error returns in call_media_endpoint,
            # which already return a dictionary with error_detail.
            # If media_response is None or not a dict, it's caught by the first check.
            print(f"{log_prefix} Falló la llamada al media endpoint o se recibió una respuesta inesperada (no dict).")
            final_content_for_ai += f"\n[Información de medio adjunto (Tipo: {detected_media_type}): No se pudo obtener información adicional o hubo un error en el endpoint.]"


    # 4. Preparar el prompt para la IA (Partes comunes)
    timezone_str = "Zona horaria desconocida" # Default fallback for timezone_str variable
    try:
        timezone_str = get_timezone_from_country(pais) # Get the pytz timezone string
        tz = pytz.timezone(timezone_str)
        now_local = datetime.now(tz)
    except (pytz.UnknownTimeZoneError, Exception) as e:
        print(f"{log_prefix} Error al obtener/usar zona horaria para '{pais}': {e}. Usando UTC para la hora de referencia.")
        # Fallback to UTC if timezone lookup or usage fails
        now_local = datetime.now(pytz.utc)
        timezone_str = "UTC" # Update timezone_str variable to reflect fallback to UTC


    # Add debug prints here to see the state before formatting
    print(f"{log_prefix} Debug: now_local.isoformat() = {now_local.isoformat()}")
    print(f"{log_prefix} Debug: timezone_str variable = {timezone_str}")


    # --- Formato de Fecha y Hora Específico para el Prompt (SECCIÓN CORREGIDA NUEVAMENTE) ---
    # Try to format using the local timezone and locale
    try:
        localized_day_name = babel.dates.format_datetime(now_local, format='EEEE', locale=idioma)
        localized_day_name = localized_day_name.capitalize()
    except babel.core.UnknownLocaleError:
        print(f"{log_prefix} Advertencia: Idioma '{idioma}' no reconocido por Babel para nombre del día. Usando 'en'.")
        localized_day_name = babel.dates.format_datetime(now_local, format='EEEE', locale='en').capitalize()
    except Exception as e:
         print(f"{log_prefix} Error formatting day name with Babel for locale '{idioma}': {e}. Using simple weekday name.")
         localized_day_name = now_local.strftime('%A') # Use local time for strftime

    try:
        localized_date_part = babel.dates.format_date(now_local, format='long', locale=idioma)
    except babel.core.UnknownLocaleError:
        print(f"{log_prefix} Advertencia: Idioma '{idioma}' no reconocido por Babel para formato de fecha. Usando 'en'.")
        localized_date_part = babel.dates.format_date(now_local, format='long', locale='en')
    except Exception as e:
         print(f"{log_prefix} Error formatting date part with Babel for locale '{idioma}': {e}. Using default strftime date part.")
         localized_date_part = now_local.strftime('%d/%m/%Y') # Use local time for strftime

    # --- Corrected Time Part Formatting using hh:mm AM/PM format (Attempt 5) ---
    formatted_time_part = ""
    try:
        # Format time in hh:mm AM/PM format (without seconds, as requested)
        # Use %I for 12-hour (01-12), %M for minute (00-59), %p for AM/PM
        time_12h_format = now_local.strftime('%I:%M %p')
        # Ensure AM/PM are capitalized as requested
        formatted_time_part = time_12h_format.replace('am', 'AM').replace('pm', 'PM')

        # Explicitly add the timezone string obtained earlier in parentheses
        # This uses the timezone_str variable directly, which should be the pytz string (e.g., "America/Bogota") or "UTC" on error
        formatted_time_part += f" ({timezone_str})"

    except Exception as e:
        print(f"{log_prefix} Error general al formatear la hora para el prompt: {e}. Usando formato simple hh:mm AM/PM.")
        # Fallback to basic hh:mm AM/PM time on error and add a placeholder for timezone
        formatted_time_part = now_local.strftime('%I:%M %p').replace('am', 'AM').replace('pm', 'PM')
        formatted_time_part += " (Error Zona Horaria)" # Use a generic error placeholder


    # Final formatted string for the prompt
    # This string will be inserted into the prompt template
    formatted_datetime_string_for_prompt = f"Es el dia {localized_day_name} {localized_date_part} y son las {formatted_time_part}"
    print(f"{log_prefix} Hora local formateada para prompt (para IA): {formatted_datetime_string_for_prompt}")

    # --- Fin Formato de Fecha y Hora Específico para el Prompt ---


    example_json_structure = """{
    "1": {
        "tipo": "mensaje",
        "mensaje": "Hola, ¿en qué puedo ayudarte?"
    },
    "2": {
        "tipo": "imagen",
        "ruta2": "URL_O_RUTA_DE_LA_IMAGEN",
        "mensaje": "Aquí tienes una imagen relevante."
    }
}"""

    # --- Estructura base del prompt (usando placeholders para .format()) ---
    prompt_template = """{promt_base}
_________________________________________
Historial de la conversación reciente (últimos {num_mensajes} mensajes del cliente):
{historial_texto}
_________________________________________
Mensaje ACTUAL del cliente (puede incluir contenido de varios mensajes cortos, **y la información de medio procesada si aplica**):
{mensaje_cliente}
_________________________________________
Fecha y hora actual ({pais}, Zona horaria: {zona_horaria}):
{fecha_hora_formateada}
_________________________________________
Instrucciones de formato de respuesta:
Genera SOLAMENTE un objeto JSON válido. La estructura debe ser un diccionario donde las claves son números secuenciales como strings ("1", "2", etc.) y los valores son objetos que describen la acción a realizar.
Cada objeto de acción DEBE tener una clave "tipo" (ej: "mensaje", "imagen", "audio", "video", "pdf", "ubicacion").
Campos adicionales dependen del tipo:
- "mensaje": requiere "mensaje" (string).
- "imagen": requiere "ruta2" (URL/path), opcional "mensaje" (caption), opcional "nombrearchivo".
- "video": requiere "ruta2" (URL/path), opcional "mensaje" (caption).
- "audio": requiere "ruta2" (URL/path) (indica si es PTT si es necesario en el envío).
- "pdf": requiere "ruta2" (URL/path), "nombrearchivo", opcional "mensaje" (caption).
- "ubicacion": requiere "lat" (float), "long" (float).

Ejemplo de estructura JSON de respuesta esperada:
{example_json_structure_placeholder}

Importante: Condensa la respuesta tanto como sea posible en el primer mensaje ("1"), pero usa claves adicionales ("2", "3", ...) if necessary to separate actions (ej: enviar una imagen y luego un texto). NO incluyas explicaciones fuera del JSON. Tu respuesta DEBE ser únicamente el objeto JSON.

Considera la conversación y el contexto proporcionado para generar la respuesta más adecuada.

---
**INSTRUCCIÓN ADICIONAL:** Responde ÚNICAMENTE en el idioma especificado: **{idioma_respuesta}**. Asegúrate de que todo el contenido de los mensajes dentro del JSON esté en **{idioma_respuesta}**.
"""
    # --- Fin Estructura base del prompt ---

    # Diccionario base con los datos comunes Y el placeholder para el JSON de ejemplo
    prompt_data_base = {
        "promt_base": req.promt,
        "num_mensajes": req.numerodemensajes,
        "historial_texto": plain_history if plain_history else "(No hay historial reciente)",
        "mensaje_cliente": final_content_for_ai, # Use the potentially modified content here
        "zona_horaria": timezone_str, # This will show the determined timezone string, like "America/Bogota" or "UTC"
        "pais": pais,
        "fecha_hora_formateada": formatted_datetime_string_for_prompt, # This now contains "Es el dia ... y son las hh:mm AM/PM (Zona Horaria)"
        "idioma_respuesta": idioma,
        "example_json_structure_placeholder": example_json_structure
    }

    # Formatear el template con los datos base para obtener el prompt original completo
    original_prompt_text = prompt_template.format(**prompt_data_base)


    # Log the original prompt (Keep this uncommented as requested)
    print(f"\n--- PRUEBA DEBUG: PROMPT ORIGINAL para {log_prefix} ---")
    print(original_prompt_text)
    print("------------------------------------------------------------\n")


    # 5. Call Gemini API with multiple retry strategies

    # USE THE ai_model PARAMETER HERE
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{ai_model}:generateContent?key={req.apikey}"

    generation_config = {
        "temperature": req.temperature,
        "topP": req.topP,
        "maxOutputTokens": req.maxOutputTokens,
        "responseMimeType": "application/json"
    }

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
    ]

    ai_response_json = None
    last_error_type = None # Track the type of the last error for conditional retries

    # --- Strategy 1: Standard retries with original prompt ---
    max_standard_retries = 2 # Total 3 attempts (0, 1, 2)
    standard_retry_delay = 2 # seconds

    for attempt in range(max_standard_retries + 1):
        current_prompt_text = original_prompt_text
        print(f"{log_prefix} Llamando a Gemini (Modelo: {ai_model}, Estrategia 1, Intento {attempt + 1}/{max_standard_retries + 1})...")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(gemini_url, json={"contents": [{"parts": [{"text": current_prompt_text}]}], "generationConfig": generation_config, "safetySettings": safety_settings}, timeout=90.0)

            response.raise_for_status() # Raise for 4xx/5xx

            gemini_output = response.json()

            # Validar si hay candidatos y contenido
            if 'candidates' in gemini_output and gemini_output['candidates']:
                content = gemini_output['candidates'][0].get('content', {})
                if 'parts' in content and content['parts']:
                    raw_text = content['parts'][0].get('text', '')
                    # print(f"{log_prefix} Respuesta cruda Gemini (Intento {attempt + 1}): {raw_text}")
                    try:
                        ai_response_json = json.loads(raw_text)
                        # Validar estructura básica. Si falla, se considera un error de formato JSON que puede requerir reintento.
                        if not isinstance(ai_response_json, dict) or not all(isinstance(v, dict) and 'tipo' in v for v in ai_response_json.values()):
                             print(f"{log_prefix} Advertencia: JSON de Gemini no tiene la estructura esperada (Intento {attempt + 1}).")
                             last_error_type = "json_structure" # Mark error type
                             # This is NOT a success, we don't break the loop here.

                        else: # Structure seems correct, it's a success
                             print(f"{log_prefix} Respuesta JSON parseada (Intento {attempt + 1}): {ai_response_json}")
                             last_error_type = None # Reset error type on success
                             break # Exit standard loop, success!


                    except json.JSONDecodeError:
                        print(f"{log_prefix} Error: JSON inválido de Gemini (Intento {attempt + 1}): {raw_text}")
                        last_error_type = "json_decode" # Mark error type

                    except ValueError as ve: # Catch ValueError from structure validation
                         print(f"{log_prefix} Error: {ve} (Intento {attempt + 1})")
                         last_error_type = "json_structure" # Mark error type


                else: # 'parts' missing
                     print(f"{log_prefix} Error: No se encontraron 'parts' en la respuesta (Intento {attempt + 1}).")
                     last_error_type = "no_parts" # Mark error type


            else: # 'candidates' missing or blocked
                 block_reason = gemini_output.get('promptFeedback', {}).get('blockReason', 'Unknown')
                 print(f"{log_prefix} Error: No 'candidates' o bloqueada (Intento {attempt + 1}). Razón: {block_reason}")
                 last_error_type = "blocked" if 'promptFeedback' in gemini_output else "no_candidates"
                 # Si bloqueada, no reintentar en este bucle
                 if last_error_type == "blocked":
                      break # Exit standard loop

        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            print(f"{log_prefix} Error HTTP {status_code} llamando a Gemini (Intento {attempt + 1}): {e.response.text}")
            last_error_type = f"http_{status_code}" # Mark error type
            # Only retry 5xx in this loop
            if status_code < 500: # Do not retry 4xx errors in this loop
                 break # Exit standard loop


        except httpx.RequestError as e:
            print(f"{log_prefix} Error de red llamando a Gemini (Intento {attempt + 1}): {e}")
            last_error_type = "network" # Mark error type

        except Exception as e:
            print(f"{log_prefix} Error inesperado durante la llamada a Gemini (Intento {attempt + 1}): {e}")
            import traceback
            traceback.print_exc()
            last_error_type = "unexpected" # Mark error type
            break # Exit standard loop for unexpected errors

        # Si llegamos aquí, ocurrió un error y no salimos del bucle
        if attempt < max_standard_retries:
             print(f"{log_prefix} Reintentando (Estrategia 1) en {standard_retry_delay} segundos...")
             await asyncio.sleep(standard_retry_delay)
        else:
             print(f"{log_prefix} Agotados reintentos (Estrategia 1). Último error: {last_error_type}")


    # --- Estrategia 2: Reintentar con prompt modificado si el error fue de formato/estructura JSON ---
    modified_prompt_retries = 1 # Total 2 intentos (0, 1) en esta fase
    modified_retry_delay = 3 # segundos (quizás un poco más)

    # Solo intentar Estrategia 2 si Estrategia 1 falló y el último error fue de formato/estructura JSON
    if ai_response_json is None and last_error_type in ["json_decode", "json_structure", "no_parts"]:
        print(f"{log_prefix} Estrategia 1 falló con error de formato/estructura ({last_error_type}). Intentando Estrategia 2 (prompt modificado).")

        # Construir el prompt modificado
        # Insertar una instrucción al principio pidiendo que reinterprete el mensaje del cliente
        modified_instruction_prefix = f"""The previous attempt to process the user's message resulted in an invalid response format. Please disregard any previous formatting issues. Carefully re-interpret the user's message and provide a valid JSON response based on the following: """ # Explicit instruction

        # Crear un nuevo diccionario para los datos del prompt modificado, basado en los datos base
        modified_prompt_data = prompt_data_base.copy() # Usar datos base
        # Modificar el campo 'mensaje_cliente' para incluir la instrucción y el mensaje original
        modified_prompt_data["mensaje_cliente"] = f"{modified_instruction_prefix}\n{final_content_for_ai}" # Usar el contenido potencialmente modificado aquí también

        # Generar el prompt modificado usando la plantilla y los datos modificados
        modified_prompt_text = prompt_template.format(**modified_prompt_data)


        # Log the modified prompt for debugging
        # print(f"\n--- PRUEBA DEBUG: PROMPT MODIFICADO para {log_prefix} ---")
        # print(modified_prompt_text)
        # print("-----------------------------------------------------------------\n")


        for attempt in range(modified_prompt_retries + 1):
             print(f"{log_prefix} Llamando a Gemini (Modelo: {ai_model}, Estrategia 2, Intento {attempt + 1}/{modified_prompt_retries + 1})...")
             try:
                async with httpx.AsyncClient() as client:
                    # Usar el prompt modificado para esta estrategia
                    response = await client.post(gemini_url, json={"contents": [{"parts": [{"text": modified_prompt_text}]}], "generationConfig": generation_config, "safety_settings": safety_settings}, timeout=90.0) # Usar safety_settings if es necesario

                response.raise_for_status() # Levantar error para 4xx/5xx

                gemini_output = response.json()

                # Validar si hay candidatos y contenido
                if 'candidates' in gemini_output and gemini_output['candidates']:
                    content = gemini_output['candidates'][0].get('content', {})
                    if 'parts' in content and content['parts']:
                        raw_text = content['parts'][0].get('text', '')
                        # print(f"{log_prefix} Respuesta cruda Gemini (Estrategia 2, Intento {attempt + 1}): {raw_text}")
                        try:
                            ai_response_json = json.loads(raw_text)
                            # Validar estructura básica
                            if not isinstance(ai_response_json, dict) or not all(isinstance(v, dict) and 'tipo' in v for v in ai_response_json.values()):
                                print(f"{log_prefix} Advertencia: JSON de Gemini no tiene la estructura esperada (Estrategia 2, Intento {attempt + 1}).")
                                last_error_type = "json_structure_mod" # Marcar tipo de error específico

                            else: # Structure seems correct, it's a success!
                                print(f"{log_prefix} Respuesta JSON parseada (Estrategia 2, Intento {attempt + 1}): {ai_response_json}")
                                last_error_type = None # Reiniciar tipo de error en caso de éxito
                                break # Salir del bucle de estrategia 2

                        except json.JSONDecodeError:
                            print(f"{log_prefix} Error: JSON inválido de Gemini (Estrategia 2, Intento {attempt + 1}): {raw_text}")
                            last_error_type = "json_decode_mod" # Marcar tipo de error específico

                        except ValueError as ve:
                            print(f"{log_prefix} Error: {ve} (Estrategia 2, Intento {attempt + 1})")
                            last_error_type = "json_structure_mod" # Marcar tipo de error específico


                    else: # Faltan 'parts'
                        print(f"{log_prefix} Error: No se encontraron 'parts' en la respuesta (Estrategia 2, Intento {attempt + 1}).")
                        last_error_type = "no_parts_mod" # Marcar tipo de error específico


                else: # Faltan 'candidates' o está bloqueada
                    block_reason = gemini_output.get('promptFeedback', {}).get('blockReason', 'Unknown')
                    print(f"{log_prefix} Error: No 'candidates' o bloqueada (Estrategia 2, Intento {attempt + 1}). Razón: {block_reason}")
                    last_error_type = "blocked_mod" if 'promptFeedback' in gemini_output else "no_candidates_mod"
                    # Si bloqueada, no reintentar en este bucle tampoco
                    if last_error_type == "blocked_mod":
                        break # Salir del bucle de estrategia 2

             except httpx.HTTPStatusError as e:
                 status_code = e.response.status_code
                 print(f"{log_prefix} Error HTTP {status_code} llamando a Gemini (Estrategia 2, Intento {attempt + 1}): {e.response.text}")
                 last_error_type = f"http_{status_code}_mod" # Marcar tipo de error específico
                 # Solo reintentar 5xx en este bucle también
                 if status_code < 500: # No reintentar errores 4xx en este bucle
                      break # Salir del bucle de estrategia 2

             except httpx.RequestError as e:
                 print(f"{log_prefix} Error de red llamando a Gemini (Estrategia 2, Intento {attempt + 1}): {e}")
                 last_error_type = "network_mod" # Marcar tipo de error específico

             except Exception as e:
                 print(f"{log_prefix} Error inesperado durante la llamada a Gemini (Estrategia 2, Intento {attempt + 1}): {e}")
                 import traceback
                 traceback.print_exc()
                 last_error_type = "unexpected_mod" # Marcar tipo de error específico
                 break # Salir del bucle de estrategia 2 para errores inesperados

             # Si llegamos aquí, ocurrió un error en la Estrategia 2 y no salimos del bucle
             if attempt < modified_prompt_retries:
                 print(f"{log_prefix} Reintentando (Estrategia 2) en {modified_retry_delay} segundos...")
                 await asyncio.sleep(modified_retry_delay)
             else:
                 print(f"{log_prefix} Agotados reintentos (Estrategia 2). Último error: {last_error_type}")


    # --- Fin del bucle de Estrategia 2 ---

    # --- Fallback Final ---
    if ai_response_json is None:
         print(f"{log_prefix} Todas las estrategias y reintentos fallaron. Preparando respuesta de fallback.")
         # Lógica de fallback
         fallback_message_content = f"Lo siento, no pude obtener una respuesta adecuada en este momento. Por favor, intenta de nuevo más tarde. ({idioma})" if idioma else "Sorry, I couldn't get a proper response at the moment. Please try again later."

         # Optional: try to be more specific based on last_error_type
         if last_error_type and "blocked" in last_error_type:
              fallback_message_content = f"Lo siento, tu solicitud fue bloqueada por el filtro de seguridad de la IA. Por favor, reformula tu mensaje. ({idioma})" if idioma else "Sorry, your request was blocked by the AI's safety filter. Please rephrase your message."
         elif last_error_type and "json" in last_error_type:
              fallback_message_content = f"Lo siento, la IA tuvo problemas para formatear la respuesta correctamente. Por favor, intenta de nuevo. ({idioma})" if idioma else "Sorry, the AI had trouble formatting the response correctly. Please try again."
         elif last_error_type and "http_4" in last_error_type:
               fallback_message_content = f"Lo siento, hubo un problema con la autenticación o la solicitud a la IA ({last_error_type}). Contacta al administrador. ({idioma})" if idioma else f"Sorry, there was an issue with the AI request ({last_error_type}). Please contact the administrator."
         elif last_error_type and "network" in last_error_type:
              fallback_message_content = f"Lo siento, hubo un problema de conexión con el servicio de IA. Verifica tu conexión. ({idioma})" if idioma else "Connection error with the AI service. Please check your connection."
         # For 5xx or no_parts or unexpected, the general message may suffice.


         ai_response_json = {"1": {"tipo": "mensaje", "mensaje": fallback_message_content}}

    # ... Resto de la función process_message_final (guardar historial, etc.) ...

    # 6. Actualizar historial con el mensaje COMBINADO y la respuesta
    # Este bloque ya maneja el caso en que ai_response_json sea None
    if ai_response_json: # Solo actualizamos si obtuvimos una respuesta (incluso si es un error/fallback)
        try:
            # hist_file_path is defined globally or in an accessible scope now
            max_cli_idx = 0
            if hist:
                numeric_cli_keys = sorted([int(k.split('_')[-1]) for k in hist if k.startswith('cliente_m_') and k.split('_')[-1].isdigit()])
                max_cli_idx = max(numeric_cli_keys + [0])

            next_cli_idx = max_cli_idx + 1

            hist[f"cliente_m_{next_cli_idx}"] = final_content_for_ai # Guardar el mensaje combinado con info de medio

            response_str_parts = []
            try:
                 response_keys_sorted = sorted(ai_response_json.keys(), key=int)
            except ValueError:
                 response_keys_sorted = sorted(ai_response_json.keys())


            for key in response_keys_sorted:
                item = ai_response_json[key]
                tipo = item.get("tipo", "desconocido")
                content = item.get("mensaje")
                if content is None:
                     content = item.get("ruta2")
                if content is None:
                     content = item.get("nombrearchivo")
                if content is None:
                     # Si el mensaje está vacío pero el tipo es algo como imagen/video con URL,
                     # solo indicar el tipo y la clave.
                     if tipo != "mensaje" and (item.get("ruta2") or item.get("nombrearchivo")):
                         content = f"({tipo} adjunto)"
                     else:
                         content = "(sin contenido)"


                # Verificar si el contenido es uno de los mensajes de error de fallback o si el último error fue un HTTP 4xx (problema de autenticación/solicitud)
                is_fallback_message = False
                possible_fallback_prefixes = [
                    (f"Lo siento, no pude obtener una respuesta adecuada en este momento" if idioma else "Sorry, I couldn't get a proper response at the moment"),
                    (f"Lo siento, tu solicitud fue bloqueada" if idioma else "Sorry, your request was blocked"),
                    (f"Lo siento, la IA tuvo problemas para formatear la respuesta" if idioma else "Sorry, the AI had trouble formatting the response"),
                     (f"Lo siento, hubo un problema con la autenticación" if idioma else "Sorry, there was an issue with the AI request"), # Check for this specific http_4xx message
                 ]
                for prefix in possible_fallback_prefixes:
                     if isinstance(content, str) and content.startswith(prefix):
                          is_fallback_message = True
                          break

                # También considerar http_4xx como un fallo claro incluso si el mensaje no es exacto
                if is_fallback_message or (last_error_type and "http_4" in last_error_type):
                     response_str_parts.append(f"ERROR_FALLBACK ({last_error_type if last_error_type else 'unknown'}): {content}")
                else:
                     response_str_parts.append(f"{tipo}: {content}")


            hist[f"respuesta_{next_cli_idx}"] = " | ".join(response_str_parts)

            # Guardar el historial actualizado en la nueva ruta userbot/phone.txt
            history_path = hist_file_path(userbot, phone) # Re-get path for clarity/safety
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(hist, f, ensure_ascii=False, indent=4)
            print(f"{log_prefix} Historial actualizado en {history_path}")

        except Exception as e:
            print(f"{log_prefix} Error al actualizar archivo de historial {history_path}: {e}")

    return ai_response_json # Retorna la respuesta JSON de la IA (éxito o fallback)


# La función send_whatsapp ya fue actualizada en respuestas anteriores para usar userbot/phone
async def send_whatsapp(server: str, userbot: str, token: str, phone: str, payload: Dict[str, Any]):
    """Envía mensajes a WhatsApp según el payload JSON de la IA."""
    # La variable 'server' probablemente contiene la URL base para el servidor principal de WA (ej: http://100.42.185.2:8012).
    # Los endpoints son /chats/send-presence?id={userbot} y /chats/send?id={userbot}
    present_url = f"{server}/chats/send-presence?id={userbot}"
    send_url = f"{server}/chats/send?id={userbot}"
    headers = {"Content-Type": "application/json", "x-access-token": token}
    log_prefix = f"[{userbot}/{phone}]"

    print(f"{log_prefix} Preparando para enviar respuesta a WhatsApp: {payload}") # Log con userbot/phone

    async with httpx.AsyncClient() as client:
        try:
            # Enviar presencia 'composing' antes de empezar a enviar mensajes
            print(f"{log_prefix} Enviando presencia 'composing'...")
            await client.post(present_url, headers=headers, json={"receiver": phone, "presence": "composing", "isGroup": False}, timeout=10.0)
            print(f"{log_prefix} Presencia 'composing' enviada.")
        except Exception as e:
            print(f"{log_prefix} Advertencia: No se pudo enviar presencia 'composing': {e}")

        try:
             payload_keys_sorted = sorted(payload.keys(), key=int)
        except ValueError:
             payload_keys_sorted = sorted(payload.keys())

        for key in payload_keys_sorted:
            item = payload[key]
            tipo = item.get("tipo")
            body = {"receiver": phone, "isGroup": False} # Asumiendo isGroup=False para chats individuales
            message_to_send = {}
            presence_before_each_message = None # Puede ser recording para audio

            try:
                if tipo == "mensaje":
                    text = item.get("mensaje")
                    if text:
                         message_to_send = {"message": {"text": text}}
                         print(f"{log_prefix} Preparando mensaje de texto: '{text}'")
                    else:
                         print(f"{log_prefix} Saltando envío de mensaje 'mensaje' vacío.")
                         continue

                elif tipo == "imagen":
                    image_url = item.get("ruta2")
                    if image_url:
                         message_part = {"image": {"url": image_url}}
                         caption = item.get("mensaje")
                         if caption: message_part["caption"] = caption
                         message_to_send = {"message": message_part}
                         print(f"{log_prefix} Preparando para enviar imagen: {image_url}")
                    else:
                         print(f"{log_prefix} Saltando envío de imagen sin URL.")
                         continue

                elif tipo == "video":
                     video_url = item.get("ruta2")
                     if video_url:
                          message_part = {"video": {"url": video_url}}
                          caption = item.get("mensaje")
                          if caption: message_part["caption"] = caption
                          message_to_send = {"message": message_part}
                          presence_before_each_message = "recording" # Establecer presencia a recording antes de enviar audio
                          print(f"{log_prefix} Preparando para enviar video: {video_url}")
                     else:
                          print(f"{log_prefix} Saltando envío de video sin URL.")
                          continue

                elif tipo == "audio":
                     audio_url = item.get("ruta2")
                     if audio_url:
                          is_ptt = item.get("ptt", True) # Asumir PTT a menos que se especifique lo contrario
                          message_part = {"audio": {"url": audio_url}, "ptt": is_ptt}
                          message_to_send = {"message": message_part}
                          presence_before_each_message = "recording" # Establecer presencia a recording antes de enviar audio
                          print(f"{log_prefix} Preparando para enviar audio (PTT: {is_ptt}): {audio_url}")
                     else:
                          print(f"{log_prefix} Saltando envío de audio sin URL.")
                          continue

                elif tipo == "pdf":
                    pdf_url = item.get("ruta2")
                    filename = item.get("nombrearchivo", "document.pdf")
                    if pdf_url:
                         message_part = {"document": {"url": pdf_url, "mimetype": "application/pdf", "fileName": filename}}
                         caption = item.get("mensaje")
                         if caption: message_part["caption"] = caption
                         message_to_send = {"message": message_part}
                         print(f"{log_prefix} Preparando para enviar PDF: {pdf_url}")
                    else:
                         print(f"{log_prefix} Saltando envío de PDF sin URL.")
                         continue

                elif tipo == "ubicacion":
                    lat = item.get("lat")
                    lon = item.get("long")
                    if lat is not None and lon is not None:
                         try:
                              lat_f = float(lat)
                              lon_f = float(lon)
                              message_to_send = {"message": {"location": {"degreesLatitude": lat_f, "degreesLongitude": lon_f}}}
                              print(f"{log_prefix} Preparando para enviar ubicación: Lat={lat_f}, Lon={lon_f}")
                         except (ValueError, TypeError):
                              print(f"{log_prefix} Saltando envío de ubicación con lat/long inválido: {lat}, {lon}")
                              continue
                    else:
                         print(f"{log_prefix} Saltando envío de ubicación sin lat/long.")
                         continue
                else:
                    print(f"{log_prefix} Advertencia: Tipo de mensaje desconocido '{tipo}' recibido de la IA. Saltando.")
                    continue

                # Enviar presencia específica (como recording) antes de enviar el mensaje si es necesario
                if presence_before_each_message:
                    try:
                        print(f"{log_prefix} Enviando presencia '{presence_before_each_message}'...")
                        await client.post(present_url, headers=headers, json={**body, "presence": presence_before_each_message}, timeout=10.0)
                        print(f"{log_prefix} Presencia '{presence_before_each_message}' enviada.")
                    except Exception as e:
                        print(f"[{userbot}/{phone}] Advertencia: No se pudo enviar presencia '{presence_before_each_message}': {e}")


                print(f"{log_prefix} Enviando a {send_url} con body: {json.dumps({**body, **message_to_send})}")
                response = await client.post(send_url, headers=headers, json={**body, **message_to_send}, timeout=30.0)
                response.raise_for_status()
                print(f"{log_prefix} Mensaje tipo '{tipo}' enviado exitosamente. Respuesta: {response.json()}")

                # Send presence back to composing after sending an audio message
                if presence_before_each_message == "recording":
                     try:
                          print(f"{log_prefix} Enviando presencia 'composing' después del audio...")
                          await client.post(present_url, headers=headers, json={**body, "presence": "composing", "isGroup": False}, timeout=10.0)
                          print(f"{log_prefix} Presencia 'composing' enviada después del audio.")
                     except Exception as e:
                          print(f"[{userbot}/{phone}] Advertencia: No se pudo enviar presencia 'composing' después del audio: {e}")


                # Add a small delay between sending multiple messages in the payload
                if payload_keys_sorted and key != payload_keys_sorted[-1]:
                     await asyncio.sleep(1.5)

            except httpx.HTTPStatusError as e:
                print(f"{log_prefix} Error HTTP {e.response.status_code} al enviar mensaje tipo '{tipo}': {e.response.text}")
            except httpx.RequestError as e:
                print(f"{log_prefix} Error de red al enviar mensaje tipo '{tipo}': {e}")
            except Exception as e:
                print(f"{log_prefix} Error inesperado al enviar mensaje tipo '{tipo}': {e}")
                import traceback
                traceback.print_exc()

        # Send 'paused' presence at the very end after all messages are sent or attempted
        try:
            print(f"{log_prefix} Enviando presencia 'paused' al finalizar...")
            await client.post(present_url, headers=headers, json={**body, "presence": "paused", "isGroup": False}, timeout=10.0)
            print(f"{log_prefix} Presencia 'paused' enviada.")
        except Exception as e:
            print(f"{log_prefix} Advertencia: No se pudo enviar presencia 'paused' al finalizar: {e}")


# La tarea de procesamiento retrasado ya fue actualizada
async def delayed_processing_task(task_key: str):
    """Tarea que se ejecuta después del retraso para procesar mensajes agrupados."""

    # Verificar nuevamente si la clave de la tarea aún existe antes de continuar después del sleep
    if task_key not in processing_tasks:
        print(f"[{task_key}] Tarea cancelada antes de iniciar el procesamiento final.")
        return

    task_info = processing_tasks[task_key]
    original_req = task_info['original_request']
    delay_seconds = original_req.delay_seconds
    phone = original_req.lineaWA
    userbot = original_req.userbot
    log_prefix = f"[{userbot}/{phone}]"

    print(f"{log_prefix} Iniciando retraso de {delay_seconds} segundos...")

    try:
        await asyncio.sleep(delay_seconds)

        # Antes de procesar, verificar el estado de pausa FINAL después de la lógica en handle_incoming_message
        # Cargar el estado más reciente del archivo
        current_state = load_contact_state(userbot, phone)

        # is_contact_paused handles the timeout check based on the loaded state
        if is_contact_paused(current_state, original_req.pause_timeout_minutes):
             print(f"{log_prefix} Contacto se pausó/sigue pausado después del retraso. Omitiendo procesamiento final.")
             # Limpiar la tarea ya que no estamos procesando
             current_task = asyncio.current_task()
             if task_key in processing_tasks and processing_tasks[task_key]['task'] is current_task:
                  del processing_tasks[task_key]
             return

        if task_key not in processing_tasks:
             print(f"{log_prefix} La tarea fue cancelada o ya no existe antes de procesar.")
             return

        task_info = processing_tasks[task_key]
        original_req = task_info['original_request']
        fragments = task_info['fragments']

        # Combinar fragmentos usando ", " como separador
        full_message = ", ".join(fragments).strip()

        # Si el mensaje, después de eliminar espacios, está vacío
        if not full_message: # Verificación simplificada como se discutió previamente
            print(f"{log_prefix} No hay mensaje para procesar después del retraso (solo espacios o vacío).")
            if task_key in processing_tasks and processing_tasks[task_key]['task'] is asyncio.current_task():
                 del processing_tasks[task_key]
            return

        print(f"{log_prefix} Retraso completado. Mensaje combinado: '{full_message}'. Iniciando procesamiento y envío...")

        # Llamar a process_message_final con la solicitud original y el mensaje combinado
        ai_response = await process_message_final(
            original_req,
            full_message,
            original_req.pais,
            original_req.idioma,
            original_req.ai_model # Pass the AI model
        )

        # Enviar la respuesta de la IA a WhatsApp si existe
        if ai_response:
            # Asegurarse de usar los datos de la solicitud original para el envío
            await send_whatsapp(
                server=original_req.server,
                userbot=original_req.userbot,
                token=original_req.token,
                phone=original_req.lineaWA, # Usar lineaWA de la solicitud original
                payload=ai_response
            )
        else:
             print(f"{log_prefix} No se obtuvo respuesta de la IA para enviar a WhatsApp.")

    except asyncio.CancelledError:
        print(f"{log_prefix} Tarea de procesamiento cancelada por nuevo mensaje.")
    except Exception as e:
        print(f"{log_prefix} Error en la tarea de procesamiento retrasado: {e}")
        import traceback
        traceback.print_exc()
    finally:
        current_task = asyncio.current_task()
        # Solo limpiar la tarea si aún es la tarea actual asociada a esta clave
        if task_key in processing_tasks and processing_tasks[task_key]['task'] is current_task:
             del processing_tasks[task_key]


@app.post("/wa/process")
async def handle_incoming_message(req: MessageRequest):
    """
    Endpoint para recibir nuevos mensajes.
    Implementa el manejo del estado de pausa por contacto.
    Si no está pausado, agrupa mensajes y programa el procesamiento con retraso.
    """
    phone = req.lineaWA
    userbot = req.userbot
    new_fragment = req.mensaje_reciente.strip()
    task_key = f"{userbot}_{phone}" # Clave para la tarea de procesamiento: userbot_lineaWA
    log_prefix = f"[{userbot}/{phone}]"

    print(f"{log_prefix} Mensaje recibido: '{req.mensaje_reciente}' (Fragmento strip: '{new_fragment}')")
    print(f"{log_prefix} Configuración recibida: País: {req.pais}, Idioma: {req.idioma}, Delay Agrupación: {req.delay_seconds}s, Timeout Pausa: {req.pause_timeout_minutes}min, Modelo IA: {req.ai_model}, Temp: {req.temperature}, TopP: {req.topP}, MaxTokens: {req.maxOutputTokens})")

    # Validation for required fields
    if not userbot:
         raise HTTPException(status_code=400, detail="El campo 'userbot' es requerido.")
    if not phone or not req.mensaje_reciente:
        raise HTTPException(status_code=400, detail="Los campos 'lineaWA' y 'mensaje_reciente' son requeridos.")
    if not req.pais:
         raise HTTPException(status_code=400, detail="El campo 'pais' es requerido.")
    if not req.idioma:
         raise HTTPException(status_code=400, detail="El campo 'idioma' es requerido.")
    # Validate ai_model is not empty (though a default is provided)
    if not req.ai_model:
         raise HTTPException(status_code=400, detail="El campo 'ai_model' es requerido.")

    # Validation for AI configuration fields
    if req.temperature is not None and (req.temperature < 0.0 or req.temperature > 1.0):
        raise HTTPException(status_code=400, detail="El campo 'temperature' debe estar entre 0.0 y 1.0")
    if req.topP is not None and (req.topP < 0.0 or req.topP > 1.0):
        raise HTTPException(status_code=400, detail="El campo 'topP' debe estar entre 0.0 y 1.0")
    if req.maxOutputTokens is not None and req.maxOutputTokens <= 0:
         raise HTTPException(status_code=400, detail="El campo 'maxOutputTokens' debe ser positivo.")

    # Validation for delay_seconds and pause_timeout_minutes
    if req.delay_seconds is not None and req.delay_seconds < 0:
         raise HTTPException(status_code=400, detail="El campo 'delay_seconds' no puede ser negativo.")
    if req.pause_timeout_minutes is not None and req.pause_timeout_minutes < 0:
         raise HTTPException(status_code=400, detail="El campo 'pause_timeout_minutes' no puede ser negativo.")

    # --- Implementación del Estado de Pausa por Contacto ---

    # 1. Cargar el estado actual del contacto
    current_state = load_contact_state(userbot, phone)
    print(f"{log_prefix} Estado inicial cargado: {current_state}")

    # 2. Obtener los timestamps de las últimas reacciones de control
    latest_pause_ts_ms, latest_unpause_ts_ms = await get_latest_control_reaction_timestamps(
        userbot=req.userbot,
        token=req.token,
        phone=req.lineaWA
    )
    print(f"{log_prefix} Timestamps de últimas reacciones de control: Pausa={latest_pause_ts_ms}, Reanudar={latest_unpause_ts_ms}")


    # 3. Determinar el nuevo estado basado en reacciones y timeout
    new_state = current_state.copy() # Start with current state

    # Timestamp de la última reacción de control registrada en el estado
    last_control_reaction_timestamp_in_state = current_state.get("last_control_reaction_timestamp", 0)

    # Timestamp de la reacción de control más reciente encontrada en la API
    abs_latest_control_ts = max(latest_pause_ts_ms, latest_unpause_ts_ms)

    # current_time_ms = int(time.time() * 1000) # Not directly used in this logic


    # --- Decisión de Estado ---

    # Prioridad 1: Reacciones manuales NUEVAS (posteriores a la que estableció el estado actual)
    if abs_latest_control_ts > last_control_reaction_timestamp_in_state:
        print(f"{log_prefix} Hay una NUEVA reacción de control ({abs_latest_control_ts}) > timestamp estado ({last_control_reaction_timestamp_in_state}). Decidiendo estado por REACCIÓN NUEVA.")
        if latest_unpause_ts_ms > latest_pause_ts_ms:
            # ✅ es la reacción de control más reciente entre las dos NUEVAS. Despausando.
            print(f"{log_prefix} La última reacción de control NUEVA es ✅. Despausando por reacción.")
            new_state["is_paused"] = False
            new_state["pause_start_time"] = None
            new_state["last_control_reaction_timestamp"] = latest_unpause_ts_ms
        else: # latest_pause_ts_ms >= latest_unpause_ts_ms
            # ✋ es la reacción de control más reciente entre las dos NUEVAS. Pausando.
            print(f"{log_prefix} La última reacción de control NUEVA es ✋. Pausando por reacción.")
            new_state["is_paused"] = True
            new_state["pause_start_time"] = datetime.now().isoformat() # Record current time as pause start
            new_state["last_control_reaction_timestamp"] = latest_pause_ts_ms

    # Prioridad 2: Timeout si el estado actual es pausado Y no hubo reacción NUEVA
    elif current_state["is_paused"] and current_state["pause_start_time"]:
        print(f"{log_prefix} No hay reacción de control NUEVA. Estado actual es pausado. Verificando timeout.")
        if req.pause_timeout_minutes > 0:
            try:
                pause_start_time = datetime.fromisoformat(current_state["pause_start_time"])
                timeout_duration = timedelta(minutes=req.pause_timeout_minutes)
                time_since_pause_start = datetime.now() - pause_start_time

                if time_since_pause_start > timeout_duration:
                    print(f"{log_prefix} Pausa automática expirada ({req.pause_timeout_minutes} minutos). Despausando por timeout.")
                    new_state["is_paused"] = False
                    new_state["pause_start_time"] = None
                    # IMPORTANTE: Cuando se despausa por timeout, NO actualizamos
                    # last_control_reaction_timestamp. Debe mantener el timestamp
                    # de la reacción ✋ que inició esta pausa ahora expirada.
                    # Esto permite que una *nueva* reacción ✋ (con timestamp > last_control_reaction_timestamp)
                    # pueda pausar nuevamente.
                else:
                     print(f"{log_prefix} Contacto sigue pausado, timeout aún activo ({time_since_pause_start} < {timeout_duration}).")
                     # El estado permanece pausado, no hay cambio en new_state.

            except ValueError:
                 print(f"[{log_prefix}] Error: pause_start_time en estado inválido '{current_state['pause_start_time']}'. Tratando como no pausado.")
                 new_state["is_paused"] = False
                 new_state["pause_start_time"] = None
                 # Keep the old last_control_reaction_timestamp.
        else:
             print(f"{log_prefix} Contacto sigue pausado (timeout_minutes es 0).")
             # El estado permanece pausado, no hay cambio en new_state.

    # Prioridad 3: Verificar si la última reacción de pausa detectada está VENCIDA por timeout,
    # SI el estado actual NO era pausado Y NO hubo una reacción NUEVA que dicte pausa.
    # Esto maneja el caso del log donde un viejo ✋ se detecta y causa pausa.
    # Solo aplicar si la última reacción de pausa detectada no es la misma que ya tenemos registrada en el estado.
    # Esto evita re-evaluar constantemente la misma reacción si el estado no ha cambiado por una NUEVA reacción.
    # También asegurar que latest_pause_ts_ms > 0 para que haya algo que evaluar.
    elif latest_pause_ts_ms > 0 and latest_pause_ts_ms != last_control_reaction_timestamp_in_state:
        print(f"{log_prefix} No hay reacción de control NUEVA. Última reacción ✋ detectada ({latest_pause_ts_ms}) != timestamp estado ({last_control_reaction_timestamp_in_state}). Verificando si esta reacción está VIGENTE por timeout.")
        if req.pause_timeout_minutes > 0:
            try:
                pause_reaction_time = datetime.fromtimestamp(latest_pause_ts_ms / 1000.0)
                time_since_pause_reaction = datetime.now() - pause_reaction_time
                timeout_duration = timedelta(minutes=req.pause_timeout_minutes)

                if time_since_pause_reaction <= timeout_duration:
                     # La última reacción de pausa detectada aún está VIGENTE dentro del timeout.
                     # Como no hubo una reacción NUEVA (punto 1), y no estábamos pausados activamente (punto 2),
                     # ESTO INDICA que la última reacción de control relevante es esta ✋ VIGENTE.
                     # Por lo tanto, debemos pausar.
                     print(f"{log_prefix} Última reacción ✋ ({latest_pause_ts_ms}) detectada está VIGENTE ({time_since_pause_reaction} <= {timeout_duration}). Pausando.")
                     new_state["is_paused"] = True
                     new_state["pause_start_time"] = datetime.now().isoformat() # Record current time as pause start
                     new_state["last_control_reaction_timestamp"] = latest_pause_ts_ms # Update timestamp with this reaction's timestamp

                else:
                     # La última reacción de pausa detectada está VENCIDA por timeout.
                     print(f"{log_prefix} Última reacción ✋ ({latest_pause_ts_ms}) detectada está VENCIDA por timeout ({time_since_pause_reaction} > {timeout_duration}). No Pausando.")
                     # El estado permanece despausado, no hay cambio en new_state.
                     # last_control_reaction_timestamp se mantiene como estaba (debería ser el de la última ✅ o 0)

            except Exception as e:
                print(f"[{log_prefix}] Error verificando vigencia de reacción ✋ antigua por timeout: {e}. No cambiando estado.")
                # El estado permanece sin cambios en caso de error.


    # Prioridad 4: Si ninguna de las condiciones anteriores se cumple, el estado se mantiene igual
    # new_state ya es current_state.copy() al inicio.

    # --- Fin Decisión de Estado ---


    # 4. Guardar el nuevo estado del contacto
    # Solo guardar si hay cambios en is_paused, pause_start_time, o last_control_reaction_timestamp
    if new_state != current_state:
        save_contact_state(userbot, phone, new_state)
        print(f"{log_prefix} Nuevo estado guardado: {new_state}")
    else:
        # Si el estado no cambió pero se está debuggeando, imprimir un mensaje
        # print(f"{log_prefix} Estado no ha cambiado. Estado actual: {current_state}")
        pass # Avoid excessive logging if state is stable.


    # 5. Verificar el estado FINAL para decidir si procesar o no
    # Usamos la función is_contact_paused con el nuevo estado para la verificación final.
    # is_contact_paused ya considera el timeout basado en el estado.
    is_currently_paused = is_contact_paused(new_state, req.pause_timeout_minutes)


    # --- Fin Implementación Estado de Pausa ---


    # Si el contacto está pausado (según el estado final determinado), omitir el procesamiento
    if is_currently_paused:
        print(f"{log_prefix} Contacto está PAUSADO. Omitiendo procesamiento automático.")
        # Si existe una tarea para este contacto, cancelarla ya que estamos omitiendo el procesamiento
        if task_key in processing_tasks:
             print(f"{log_prefix} Contacto pausado, cancelando tarea de procesamiento pendiente si existe.")
             try:
                  processing_tasks[task_key]['task'].cancel()
                  del processing_tasks[task_key] # Limpiar del diccionario inmediatamente
             except Exception as e:
                  print(f"{log_prefix} Error al intentar cancelar o eliminar tarea pendiente: {e}")

        # No continuamos con la lógica de agrupamiento ni programación de tarea
        return {"status": "paused", "detail": f"Contacto {log_prefix} pausado. Procesamiento automático omitido."}


    # Si no está pausado, proceder con la agrupación de mensajes y la lógica de tarea retrasada
    print(f"{log_prefix} Contacto no está pausado. Procediendo con el procesamiento.")

    # Verificar si ya existe una tarea pendiente para esta clave (userbot + lineaWA)
    if task_key in processing_tasks:
        print(f"{log_prefix} Ya existe una tarea pendiente. Cancelando la anterior y reiniciando temporizador.")
        existing_task_info = processing_tasks[task_key]

        # Cancelar la tarea anterior
        existing_task_info['task'].cancel()

        # Añadir el nuevo fragmento (versión sin espacios al inicio/final)
        existing_task_info['fragments'].append(new_fragment)

        # Actualizar la solicitud original con los parámetros más recientes (incluye todos los campos: userbot, phone, delay, temp, topP, maxTokens, pause_timeout_minutes, ai_model, etc.)
        existing_task_info['original_request'] = req

        # Crear y programar la NUEVA tarea (reemplaza la antigua en el diccionario)
        new_task = asyncio.create_task(delayed_processing_task(task_key))
        processing_tasks[task_key] = {
            'task': new_task,
            'fragments': existing_task_info['fragments'], # Lista actualizada
            'original_request': existing_task_info['original_request'] # Solicitud actualizada
        }
        print(f"{log_prefix} Temporizador reiniciado con delay {req.delay_seconds}s. Fragmentos acumulados: {existing_task_info['fragments']}")

    else:
        # Es el primer mensaje (o el primero después de un procesamiento completado) para este userbot+lineaWA
        print(f"{log_prefix} No hay tarea pendiente para esta clave. Creando nueva tarea con retraso de {req.delay_seconds}s.")

        # Crear y programar la tarea usando la clave de tarea
        new_task = asyncio.create_task(delayed_processing_task(task_key))

        # Almacenar la información de la tarea usando la clave de tarea
        processing_tasks[task_key] = {
            'task': new_task,
            'fragments': [new_fragment], # Iniciar la lista de fragmentos
            'original_request': req # Guardar la solicitud completa (incluye todos los campos)
        }

    return {"status": "received", "detail": f"Mensaje para {log_prefix} recibido y programado para procesamiento con delay {req.delay_seconds}s."}


@app.post("/wa/delete-history")
async def handle_delete_history(req: DeleteHistoryRequest):
    """
    Endpoint para eliminar historiales de chat.
    Permite eliminar todos los historiales de un userbot o una lista específica de lineaWAs.
    También elimina los archivos de estado de pausa asociados.
    """
    userbot = req.userbot
    userbot_hist_dir = os.path.join(HIST_BASE_DIR, userbot)
    userbot_paused_status_dir = os.path.join(PAUSED_STATUS_DIR, userbot) # Directorio de estados de pausa
    results: Dict[str, Any] = {"userbot": userbot, "deleted_count": 0, "failed_count": 0, "deleted_items": [], "failed_items": {}}
    log_prefix = f"[{userbot}]"

    if not userbot:
         raise HTTPException(status_code=400, detail="El campo 'userbot' es requerido.")

    print(f"{log_prefix} Solicitud de eliminación de historial recibida (delete_all: {req.delete_all}).")

    if req.delete_all:
        # Eliminar todo el historial para este userbot
        print(f"{log_prefix} Intentando eliminar todo el historial en: {userbot_hist_dir}")
        if os.path.isdir(userbot_hist_dir):
            try:
                # Listar todos los archivos .txt en el directorio del userbot
                files_to_delete = [f for f in os.listdir(userbot_hist_dir) if f.endswith('.txt')]
                print(f"{log_prefix} Archivos .txt encontrados para eliminar: {files_to_delete}")

                if files_to_delete:
                    for filename in files_to_delete:
                        file_path = os.path.join(userbot_hist_dir, filename)
                        try:
                            os.remove(file_path)
                            results["deleted_items"].append(f"historial/{filename}")
                            results["deleted_count"] += 1
                            print(f"{log_prefix} Eliminado historial: {file_path}")
                        except OSError as e:
                            results["failed_items"][f"historial/{filename}"] = str(e)
                            results["failed_count"] += 1
                            print(f"{log_prefix} Error al eliminar historial {file_path}: {e}")
                else:
                    print(f"[{userbot}] No se encontraron archivos .txt en '{userbot_hist_dir}'.")

            except Exception as e:
                results["status"] = "error" # Marcar como error si falla la lista de archivos
                results["message"] = f"Ocurrió un error al listar archivos de historial: {e}"
                print(f"{log_prefix} Error al listar archivos de historial: {e}")

        # Eliminar todos los archivos de estado de pausa para este userbot
        print(f"{log_prefix} Intentando eliminar todos los archivos de estado de pausa en: {userbot_paused_status_dir}")
        if os.path.isdir(userbot_paused_status_dir):
            try:
                # Listar todos los archivos .json en el directorio de estado de pausa del userbot
                files_to_delete = [f for f in os.listdir(userbot_paused_status_dir) if f.endswith('.json')]
                print(f"{log_prefix} Archivos .json de estado de pausa encontrados para eliminar: {files_to_delete}")

                if files_to_delete:
                    for filename in files_to_delete:
                        file_path = os.path.join(userbot_paused_status_dir, filename)
                        try:
                            os.remove(file_path)
                            results["deleted_items"].append(f"paused_status/{filename}")
                            results["deleted_count"] += 1
                            # También limpiar del cache
                            phone = filename.replace('.json', '')
                            if userbot in contact_pause_state_cache and phone in contact_pause_state_cache[userbot]:
                                 del contact_pause_state_cache[userbot][phone]

                            print(f"{log_prefix} Eliminado archivo de estado de pausa: {file_path}")
                        except OSError as e:
                            results["failed_items"][f"paused_status/{filename}"] = str(e)
                            results["failed_count"] += 1
                            print(f"{log_prefix} Error al eliminar archivo de estado de pausa {file_path}: {e}")
                else:
                    print(f"[{userbot}] No se encontraron archivos .json en '{userbot_paused_status_dir}'.")

            except Exception as e:
                results["status"] = "error" # Marcar como error si falla la lista de archivos
                results["message"] = f"Ocurrió un error al listar archivos de estado de pausa: {e}"
                print(f"{log_prefix} Error al listar archivos de estado de pausa: {e}")


        if "status" not in results or results["status"] != "error": # No sobrescribir un error de listado anterior
            results["status"] = "success" if results["failed_count"] == 0 else "partial_success"
            results["message"] = f"Intentada eliminación completa del historial y estado de pausa para el userbot '{userbot}'."


    else: # delete_all is False
        # Eliminar historiales específicos
        print(f"[{userbot}] Intentando eliminar historiales específicos en: {userbot_hist_dir}")
        if not req.lineaWAs_to_delete:
             raise HTTPException(status_code=400, detail="El campo 'lineaWAs_to_delete' es requerido y no puede estar vacío si 'delete_all' es false.")

        attempted_count = len(req.lineaWAs_to_delete)
        if not os.path.isdir(userbot_hist_dir):
             # Log failures for all attempted phones if directory is missing
            print(f"[{userbot}] Directorio '{userbot_hist_dir}' no encontrado para eliminación específica.")
            for phone in req.lineaWAs_to_delete:
                 filename = f"{phone}.txt"
                 results["failed_items"][f"historial/{filename}"] = "Directorio de userbot no encontrado"
                 results["failed_count"] += 1
        else:
            for phone in req.lineaWAs_to_delete:
                filename = f"{phone}.txt"
                file_path = os.path.join(userbot_hist_dir, filename)
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        results["deleted_items"].append(f"historial/{filename}")
                        results["deleted_count"] += 1
                        print(f"[{userbot}] Eliminado historial: {file_path}")
                    else:
                        results["failed_items"][f"historial/{filename}"] = "Archivo no encontrado"
                        results["failed_count"] += 1
                        print(f"[{userbot}] Archivo de historial no encontrado: {file_path}")
                except OSError as e:
                    results["failed_items"][f"historial/{filename}"] = str(e)
                    results["failed_count"] += 1
                    print(f"[{userbot}] Error al eliminar historial {file_path}: {e}")
                except Exception as e:
                    results["failed_items"][f"historial/{filename}"] = f"Error inesperado: {e}"
                    results["failed_count"] += 1
                    print(f"[{userbot}/{phone}] Error inesperado al eliminar historial {file_path}: {e}")

            # Eliminar archivos de estado de pausa específicos
            print(f"[{userbot}] Intentando eliminar archivos de estado de pausa específicos...")
            if os.path.isdir(userbot_paused_status_dir):
                for phone in req.lineaWAs_to_delete:
                     filename = f"{phone}.json"
                     file_path = os.path.join(userbot_paused_status_dir, filename)
                     try:
                          if os.path.exists(file_path):
                               os.remove(file_path)
                               results["deleted_items"].append(f"paused_status/{filename}")
                               results["deleted_count"] += 1
                               # También limpiar del cache
                               if userbot in contact_pause_state_cache and phone in contact_pause_state_cache[userbot]:
                                    del contact_pause_state_cache[userbot][phone]
                               print(f"[{userbot}] Eliminado archivo de estado de pausa: {file_path}")
                          else:
                               results["failed_items"][f"paused_status/{filename}"] = "Archivo no encontrado"
                               results["failed_count"] += 1
                               print(f"[{userbot}] Archivo de estado de pausa no encontrado: {file_path}")
                     except OSError as e:
                          results["failed_items"][f"paused_status/{filename}"] = str(e)
                          results["failed_count"] += 1
                          print(f"[{userbot}] Error al eliminar archivo de estado de pausa {file_path}: {e}")
                     except Exception as e:
                          results["failed_items"][f"paused_status/{filename}"] = f"Error inesperado: {e}"
                          results["failed_count"] += 1
                          print(f"[{userbot}/{phone}] Error inesperado al eliminar archivo de estado de pausa {file_path}: {e}")


        results["status"] = "success" if results["failed_count"] == 0 else "partial_success"
        results["message"] = f"Intentada eliminación de {attempted_count} historiales y estados de pausa específicos para el userbot '{userbot}'."
        results["attempted_count"] = attempted_count # Añadir contador de intentos para mayor claridad


    print(f"[{userbot}] Eliminación finalizada. Resultados: {results}")
    return results


# ----- Run Server (for local testing) -----
if __name__ == "__main__":
    import uvicorn
    print("Iniciando servidor FastAPI...")
    print("Visita http://127.0.0.1:8000/docs para la interfaz de Swagger UI.")
    # uvicorn.run(app, host="0.0.0.0", port=8000)