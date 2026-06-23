from fastapi import Request
from routers.bots_routes import router as bots_router
from routers.auth_routes import router as auth_router
import asyncio
import json
import os
import re  # Añadido para el endpoint de historial
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta, timezone as dt_timezone
import pytz
import babel.dates
import time

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dotenv import load_dotenv  # Para cargar variables de .env

# Importaciones de Gemini según ejemplo.py
from google import genai
from google.genai import types as genai_types

# Importar desde prompt_config.py
from prompt_config import (
    EXAMPLE_JSON_STRUCTURE,
    get_prompt_template
)

# Cargar variables de entorno del archivo .env
load_dotenv()

GEMINI_FALLBACK_MODELS_STR = os.getenv(
    "GEMINI_FALLBACK_MODELS", "gemini-2.5-flash")
GEMINI_FALLBACK_MODELS_LIST = [
    model.strip() for model in GEMINI_FALLBACK_MODELS_STR.split(',') if model.strip()]
if not GEMINI_FALLBACK_MODELS_LIST:  # Asegurar que haya al menos un modelo de fallback
    GEMINI_FALLBACK_MODELS_LIST = ["gemini-2.5-flash"]

# Lista de modelos compatibles con thinking_config
THINKING_COMPATIBLE_MODELS = ["gemini-3.1-pro-preview",
                              "gemini-3-flash-preview", "gemini-2.5-pro", "gemini-2.5-flash"]

BASE_DIR = "./Download/AutoSystem"
HIST_BASE_DIR = os.path.join(BASE_DIR, "historial")
CONF_DIR = os.path.join(BASE_DIR, "conf_2")
PAUSED_STATUS_DIR = os.path.join(BASE_DIR, "paused_status")
ORDERS_FILE_PATH = os.path.join(BASE_DIR, "orders.json")

for d in [BASE_DIR, HIST_BASE_DIR, CONF_DIR, PAUSED_STATUS_DIR]:
    os.makedirs(d, exist_ok=True)

if not os.path.exists(ORDERS_FILE_PATH):
    with open(ORDERS_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump({}, f)

COUNTRY_TIMEZONE_MAP = {
    "argentina": "America/Argentina/Buenos_Aires", "bolivia": "America/La_Paz",
    "brasil": "America/Sao_Paulo", "chile": "America/Santiago",
    "colombia": "America/Bogota", "costa rica": "America/Costa_Rica",
    "cuba": "America/Havana", "ecuador": "America/Guayaquil",
    "el salvador": "America/El_Salvador", "guatemala": "America/Guatemala",
    "honduras": "America/Tegucigalpa", "mexico": "America/Mexico_City",
    "nicaragua": "America/Managua", "panama": "America/Panama",
    "paraguay": "America/Asuncion", "peru": "America/Lima",
    "puerto rico": "America/Puerto_Rico", "republica dominicana": "America/Santo_Domingo",
    "españa": "Europe/Madrid", "uruguay": "America/Montevideo",
    "venezuela": "America/Caracas", "usa": "America/New_York",
    "canada": "America/Toronto", "uk": "Europe/London",
    "france": "Europe/Paris", "germany": "Europe/Berlin",
    "italy": "Europe/Rome", "portugal": "Europe/Lisbon",
    "australia": "Australia/Sydney", "india": "Asia/Kolkata",
    "china": "Asia/Shanghai", "japan": "Asia/Tokyo",
}


def get_timezone_from_country(country_name: str) -> str:
    return COUNTRY_TIMEZONE_MAP.get(country_name.lower(), "UTC")


def format_timestamp_with_timezone(timestamp_ms: int, pais: str) -> str:
    """
    Convierte un timestamp en milisegundos a una fecha legible usando la zona horaria del país.

    Args:
        timestamp_ms: Timestamp en milisegundos
        pais: Nombre del país para determinar la zona horaria

    Returns:
        Fecha formateada como string en formato legible
    """
    try:
        # Convertir timestamp de milisegundos a segundos
        timestamp_s = timestamp_ms / 1000

        # Crear datetime UTC desde el timestamp
        utc_dt = datetime.fromtimestamp(timestamp_s, tz=pytz.utc)

        # Obtener zona horaria del país
        timezone_str = get_timezone_from_country(pais)

        try:
            # Convertir a la zona horaria local
            local_tz = pytz.timezone(timezone_str)
            local_dt = utc_dt.astimezone(local_tz)
        except pytz.UnknownTimeZoneError:
            # Fallback a UTC si la zona horaria no es válida
            local_dt = utc_dt
            timezone_str = "UTC"

        # Formatear la fecha de manera legible
        formatted_date = local_dt.strftime('%d/%m/%Y %I:%M:%S %p')

        return f"{formatted_date} ({timezone_str})"

    except (ValueError, OSError, OverflowError) as e:
        # En caso de error, retornar el timestamp original
        return f"Timestamp: {timestamp_ms}"


class MessageRequest(BaseModel):
    lineaWA: str
    mensaje_reciente: str
    userbot: str
    apikey: str
    server: str
    numerodemensajes: int
    promt: str
    token: str
    pais: str = Field(...)
    idioma: str = Field(...)
    delay_seconds: float = Field(7.0)
    temperature: float = Field(0.5)
    topP: float = Field(0.95)
    maxOutputTokens: int = Field(4096)
    pause_timeout_minutes: int = Field(0)
    ai_model: str = Field(
        GEMINI_FALLBACK_MODELS_LIST[0] if GEMINI_FALLBACK_MODELS_LIST else "gemini-3.0-flash")
    thinking_budget: int = Field(
        0, description="Thinking budget for Gemini 2.5/3.0 models: -1 to enable reasoning, 0 to disable.")
    thinking_level: str = Field(
        "HIGH", description="Thinking level for Gemini 3.0 models: MINIMAL, LOW, DEFAULT, HIGH")
    media_resolution: str = Field(
        "MEDIA_RESOLUTION_HIGH", description="Media resolution: LOW, DEFAULT, HIGH")
    use_google_search: bool = Field(
        False, description="Activar búsqueda en Google")
    use_google_maps: bool = Field(
        False, description="Activar búsqueda en Google Maps")

    # Parámetros para construcción dinámica de prompt con estados
    estados_generales: Optional[str] = Field(
        None, description="String con contenido de estados generales para incluir en el prompt")
    estados_especificos: Optional[str] = Field(
        None, description="String con contenido de estados específicos para incluir en el prompt")

    # Nuevos campos para funcionalidad de notificaciones
    lineaogruponotificacion: Optional[str] = Field(
        None, description="Línea de WhatsApp o grupo para enviar notificaciones")
    lineaogrupo: Optional[bool] = Field(
        None, description="True si es grupo, False si es línea individual")
    estado: Optional[str] = Field(
        None, description="Estado que debe coincidir con estado_conversacion de la IA para activar notificación")
    activarnotificacion: Optional[bool] = Field(
        False, description="True para activar notificaciones, False para desactivar")

    # Nuevos campos para userbot opcional en notificaciones
    activaruserbotopcional: Optional[bool] = Field(
        False, description="True para usar un userbot diferente para notificaciones")
    userbotopcional: Optional[str] = Field(
        None, description="ID del userbot opcional para enviar notificaciones")


class DeleteHistoryRequest(BaseModel):
    userbot: str = Field(...)
    delete_all: bool = Field(...)
    lineaWAs_to_delete: Optional[List[str]] = Field(None)


class OrderStatusUpdate(BaseModel):
    status: str


class GetHistoryRequest(BaseModel):
    userbot: str
    lineaWA: str
    role_filter: Optional[str] = Field(
        None,
        description="Filter by role: 'cliente' (includes 'usuario_pausado'), 'asistente' (includes 'operador'), or specific: 'cliente', 'asistente', 'operador', 'usuario_pausado'",
        examples=["cliente", "asistente", "operador", "usuario_pausado"]
    )
    count: Optional[int] = Field(
        None,
        ge=0,
        description="Number of latest messages. 0 or None for all matching messages."
    )
    keyword_search: Optional[str] = Field(
        None,
        description="Case-insensitive regex keyword search in message content."
    )


app = FastAPI(title="WhatsApp Message Processor",
              version="1.2.31_gemini_thinking_config_v2")


app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
app.include_router(bots_router, prefix="/api/bots", tags=["bots"])

processing_tasks: Dict[str, Dict[str, Any]] = {}
contact_pause_state_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
# Cache para control de notificaciones duplicadas: { "userbot_phone": { "last_state": "...", "timestamp": 1234567890 } }
notification_state_cache: Dict[str, Dict[str, Any]] = {}


class GeminiBlockedError(Exception):
    def __init__(self, message, block_reason_name):
        super().__init__(message)
        self.block_reason_name = block_reason_name


def get_contact_state_file_path(userbot: str, phone: str) -> str:
    userbot_paused_status_dir = os.path.join(PAUSED_STATUS_DIR, userbot)
    os.makedirs(userbot_paused_status_dir, exist_ok=True)
    return os.path.join(userbot_paused_status_dir, f"{phone}.json")


def load_contact_state(userbot: str, phone: str) -> Dict[str, Any]:
    filepath = get_contact_state_file_path(userbot, phone)
    cache_key = f"{userbot}_{phone}"
    if cache_key in contact_pause_state_cache:
        return contact_pause_state_cache[cache_key].copy()

    initial_state = {
        "is_paused": False, "pause_start_time": None,
        "last_control_reaction_timestamp": 0,
        "last_message_timestamp_during_pause_ms": 0
    }
    if not os.path.exists(filepath):
        contact_pause_state_cache[cache_key] = initial_state.copy()
        return initial_state.copy()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
        for key, default_val in initial_state.items():
            state.setdefault(key, default_val)
        if not isinstance(state["is_paused"], bool):
            state["is_paused"] = False
        if state["pause_start_time"] is not None and not isinstance(state["pause_start_time"], str):
            state["pause_start_time"] = None
        if not isinstance(state["last_control_reaction_timestamp"], (int, float)):
            state["last_control_reaction_timestamp"] = 0
        if not isinstance(state["last_message_timestamp_during_pause_ms"], (int, float)):
            state["last_message_timestamp_during_pause_ms"] = 0

        contact_pause_state_cache[cache_key] = state.copy()
        return state
    except Exception as e:
        print(
            f"[{userbot}/{phone}] Error loading state {filepath}: {e}. Returning initial.")
        if cache_key in contact_pause_state_cache:
            del contact_pause_state_cache[cache_key]
        return initial_state.copy()


def save_contact_state(userbot: str, phone: str, state: Dict[str, Any]):
    filepath = get_contact_state_file_path(userbot, phone)
    cache_key = f"{userbot}_{phone}"
    try:
        current_defaults = {"is_paused": False, "pause_start_time": None,
                            "last_control_reaction_timestamp": 0, "last_message_timestamp_during_pause_ms": 0}
        for key, default_val in current_defaults.items():
            state.setdefault(key, default_val)

        if state.get("pause_start_time") and isinstance(state["pause_start_time"], str):
            try:
                datetime.fromisoformat(state["pause_start_time"])
            except ValueError:
                state["pause_start_time"] = None

        contact_pause_state_cache[cache_key] = state.copy()

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"[{userbot}/{phone}] Error saving state {filepath}: {e}")
        if cache_key in contact_pause_state_cache:
            del contact_pause_state_cache[cache_key]


def convert_timestamp_to_ms(timestamp_raw) -> int:
    """
    Convierte diferentes formatos de timestamp a milisegundos.
    Maneja tanto el formato antiguo (string/int) como el nuevo formato (objeto con low/high).
    """
    if isinstance(timestamp_raw, str) and timestamp_raw.isdigit():
        return int(timestamp_raw)
    elif isinstance(timestamp_raw, (int, float)):
        return int(timestamp_raw)
    elif isinstance(timestamp_raw, dict):
        # Nuevo formato: objeto con propiedades low, high, unsigned
        low = timestamp_raw.get('low', 0)
        high = timestamp_raw.get('high', 0)
        unsigned = timestamp_raw.get('unsigned', False)

        if isinstance(low, (int, float)) and isinstance(high, (int, float)):
            # Combinar low y high para formar un timestamp de 64 bits
            # Si low es negativo y unsigned es False, necesitamos manejarlo correctamente
            if not unsigned and low < 0:
                # Convertir low negativo a su representación unsigned de 32 bits
                low = low + (1 << 32)

            # Combinar high (32 bits altos) con low (32 bits bajos)
            timestamp_ms = (high << 32) | low
            return int(timestamp_ms)

    return 0


def extract_valid_push_name(push_name_raw: str) -> str:
    """
    Extrae y limpia un pushName válido removiendo emojis, caracteres especiales y espacios extra.

    Args:
        push_name_raw (str): El pushName crudo extraído del mensaje

    Returns:
        str: El pushName limpio y válido, o cadena vacía si no es válido
    """
    if not push_name_raw or not isinstance(push_name_raw, str):
        return ""

    # Remover emojis y caracteres especiales, mantener solo letras, números, espacios y algunos caracteres básicos
    import re
    # Patrón que mantiene letras (incluyendo acentos), números, espacios, guiones y puntos
    cleaned = re.sub(r'[^\w\s\-\.\u00C0-\u017F]', '',
                     push_name_raw, flags=re.UNICODE)

    # Remover espacios extra y limpiar
    cleaned = ' '.join(cleaned.split())

    # Validar que tenga al menos 1 carácter y no más de 50
    if len(cleaned) >= 1 and len(cleaned) <= 50:
        return cleaned

    return ""


async def get_latest_control_reaction_timestamps_and_push_name(userbot: str, token: str, phone: str, server: str) -> tuple[int, int, str]:
    """
    Obtiene los timestamps de las reacciones de control más recientes Y extrae el pushName del usuario.

    Returns:
        tuple[int, int, str]: (latest_pause_ts_ms, latest_unpause_ts_ms, user_push_name)
    """
    log_prefix = f"[{userbot}/{phone}]"
    jid = f"{phone}@s.whatsapp.net"
    reaction_api_base_url = server
    reaction_check_url_val = f"{reaction_api_base_url}/chats/{jid}?id={userbot}&cursor_fromMe=false&isGroup=false&limit=50"
    headers = {"x-access-token": token}
    latest_pause_ts_ms, latest_unpause_ts_ms = 0, 0
    user_push_name = ""

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(reaction_check_url_val, headers=headers, timeout=15.0)
            response.raise_for_status()
            response_data = response.json()
            # DEBUG LOG TO VERIFY FULL RESPONSE DATA:
            print(
                f"{log_prefix} DEBUG BAILEYS RESPONSE: Tipo: {type(response_data)}, Tiene 'data': {'data' in response_data}")
            if isinstance(response_data, dict) and 'data' in response_data and isinstance(response_data['data'], list):
                print(
                    f"{log_prefix} DEBUG BAILEYS RESPONSE: Mensajes devueltos por la API: {len(response_data['data'])}")
                for msg in response_data['data']:
                    try:
                        # DEBUG LOG TO SEE EVERY MESSAGE
                        key_info = msg.get('key', {})
                        from_me = key_info.get('fromMe', 'NO_KEY')
                        push_name_raw = msg.get('pushName', '')
                        print(
                            f"{log_prefix} DEBUG MSG: fromMe={from_me}, pushName_raw='{push_name_raw}'")

                        # Extraer pushName de mensajes del usuario (fromMe=false)
                        if not key_info.get('fromMe', False):
                            if not user_push_name and push_name_raw:
                                cleaned_push_name = extract_valid_push_name(
                                    push_name_raw)
                                if cleaned_push_name:
                                    user_push_name = cleaned_push_name
                                    print(
                                        f"{log_prefix} PushName extraído exitosamente: '{user_push_name}' (original: '{push_name_raw}')")

                        # Procesar reacciones de control
                        if 'message' in msg and 'reactionMessage' in msg['message']:
                            reaction_data = msg['message']['reactionMessage']
                            reaction_text = reaction_data.get('text')
                            reaction_ts_ms_raw = reaction_data.get(
                                'senderTimestampMs')

                            # Usar la nueva función para convertir el timestamp
                            reaction_ts_ms = convert_timestamp_to_ms(
                                reaction_ts_ms_raw)

                            # Log para debugging
                            print(
                                f"{log_prefix} Procesando reacción: texto='{reaction_text}', timestamp_raw={reaction_ts_ms_raw}, timestamp_ms={reaction_ts_ms}")

                            if not reaction_text or reaction_ts_ms <= 0:
                                continue

                            is_reacting_msg_from_bot = msg.get(
                                'key', {}).get('fromMe', False)
                            is_reacted_to_msg_from_bot = reaction_data.get(
                                'key', {}).get('fromMe', False)

                            if is_reacting_msg_from_bot and is_reacted_to_msg_from_bot:
                                if reaction_text == '✋' and reaction_ts_ms > latest_pause_ts_ms:
                                    latest_pause_ts_ms = reaction_ts_ms
                                    print(
                                        f"{log_prefix} Nueva reacción de pausa detectada: {reaction_ts_ms}")
                                elif reaction_text == '✅' and reaction_ts_ms > latest_unpause_ts_ms:
                                    latest_unpause_ts_ms = reaction_ts_ms
                                    print(
                                        f"{log_prefix} Nueva reacción de despausa detectada: {reaction_ts_ms}")
                    except Exception as e_inner:
                        print(
                            f"{log_prefix} Error processing one reaction: {e_inner} Data: {msg.get('message', {}).get('reactionMessage')}")

                return latest_pause_ts_ms, latest_unpause_ts_ms, user_push_name
            else:
                print(
                    f"{log_prefix} Reaction API unexpected structure: {response_data}")
                return 0, 0, ""
    except Exception as e:
        print(f"{log_prefix} Error fetching control reactions and push name: {e}")
        return 0, 0, ""


async def get_latest_control_reaction_timestamps(userbot: str, token: str, phone: str, server: str) -> tuple[int, int]:
    log_prefix = f"[{userbot}/{phone}]"
    jid = f"{phone}@s.whatsapp.net"
    reaction_api_base_url = server
    reaction_check_url_val = f"{reaction_api_base_url}/chats/{jid}?id={userbot}&cursor_fromMe=false&isGroup=false&limit=50"
    headers = {"x-access-token": token}
    latest_pause_ts_ms, latest_unpause_ts_ms = 0, 0
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(reaction_check_url_val, headers=headers, timeout=15.0)
            response.raise_for_status()
            response_data = response.json()
            if isinstance(response_data, dict) and 'data' in response_data and isinstance(response_data['data'], list):
                for msg in response_data['data']:
                    try:
                        if 'message' in msg and 'reactionMessage' in msg['message']:
                            reaction_data = msg['message']['reactionMessage']
                            reaction_text = reaction_data.get('text')
                            reaction_ts_ms_raw = reaction_data.get(
                                'senderTimestampMs')

                            # Usar la nueva función para convertir el timestamp
                            reaction_ts_ms = convert_timestamp_to_ms(
                                reaction_ts_ms_raw)

                            # Log para debugging
                            print(
                                f"{log_prefix} Procesando reacción: texto='{reaction_text}', timestamp_raw={reaction_ts_ms_raw}, timestamp_ms={reaction_ts_ms}")

                            if not reaction_text or reaction_ts_ms <= 0:
                                continue

                            is_reacting_msg_from_bot = msg.get(
                                'key', {}).get('fromMe', False)
                            is_reacted_to_msg_from_bot = reaction_data.get(
                                'key', {}).get('fromMe', False)

                            if is_reacting_msg_from_bot and is_reacted_to_msg_from_bot:
                                if reaction_text == '✋' and reaction_ts_ms > latest_pause_ts_ms:
                                    latest_pause_ts_ms = reaction_ts_ms
                                    print(
                                        f"{log_prefix} Nueva reacción de pausa detectada: {reaction_ts_ms}")
                                elif reaction_text == '✅' and reaction_ts_ms > latest_unpause_ts_ms:
                                    latest_unpause_ts_ms = reaction_ts_ms
                                    print(
                                        f"{log_prefix} Nueva reacción de despausa detectada: {reaction_ts_ms}")
                    except Exception as e_inner:
                        print(
                            f"{log_prefix} Error processing one reaction: {e_inner} Data: {msg.get('message', {}).get('reactionMessage')}")
                return latest_pause_ts_ms, latest_unpause_ts_ms
            else:
                print(
                    f"{log_prefix} Reaction API unexpected structure: {response_data}")
                return 0, 0
    except Exception as e:
        print(f"{log_prefix} Error fetching control reactions: {e}")
        return 0, 0


def is_contact_paused(state: Dict[str, Any], pause_timeout_minutes: int) -> bool:
    is_paused_flag = state.get("is_paused", False)
    pause_start_time_str = state.get("pause_start_time")

    if not is_paused_flag:
        return False

    if pause_start_time_str:
        if pause_timeout_minutes > 0:
            try:
                pause_start_time = datetime.fromisoformat(pause_start_time_str)
                if pause_start_time.tzinfo is None:
                    pause_start_time = pause_start_time.replace(
                        tzinfo=dt_timezone.utc)

                now_aware = datetime.now(dt_timezone.utc)
                if now_aware - pause_start_time > timedelta(minutes=pause_timeout_minutes):
                    return False
                else:
                    return True
            except ValueError:
                print(
                    f"Invalid pause_start_time format: {pause_start_time_str}. Assuming not paused due to error.")
                return False
            except Exception as e:
                print(
                    f"Error checking pause timeout: {e}. Assuming not paused due to error.")
                return False
        else:
            return True
    else:
        print(f"Warning: Contact is_paused is True but pause_start_time is missing. Treating as not effectively paused.")
        return False


def hist_file_path(userbot: str, phone: str) -> str:
    userbot_hist_dir = os.path.join(HIST_BASE_DIR, userbot)
    os.makedirs(userbot_hist_dir, exist_ok=True)
    return os.path.join(userbot_hist_dir, f"{phone}.json")


async def fetch_chat_history_since_timestamp_ms(userbot: str, token: str, phone: str, since_timestamp_ms: int, server: str) -> List[Dict[str, Any]]:
    log_prefix = f"[{userbot}/{phone}/fetch_history]"
    jid = f"{phone}@s.whatsapp.net"
    reaction_api_base_url = server
    history_url = f"{reaction_api_base_url}/chats/{jid}?id={userbot}&cursor_fromMe=false&isGroup=false&limit=100&since={since_timestamp_ms + 1}"
    headers = {"x-access-token": token}
    new_messages = []
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(history_url, headers=headers, timeout=20.0)
            response.raise_for_status()
            response_data = response.json()
            if isinstance(response_data, dict) and 'data' in response_data and isinstance(response_data['data'], list):
                messages_from_api = sorted(
                    response_data['data'],
                    key=lambda m: int(m.get("messageTimestamp", 0))
                )
                new_messages = messages_from_api
            else:
                print(
                    f"{log_prefix} History API unexpected structure: {response_data}")
    except Exception as e:
        print(f"{log_prefix} Error fetching history: {e}")
    return new_messages


def _add_or_merge_message_group(
    history_list: List[Dict[str, Any]],
    role: str,
    texts: List[str],
    timestamp_s: int,
    log_prefix: str
) -> bool:
    if not texts:
        return False

    separator = ", "

    cleaned_texts = [t.strip() for t in texts if t and t.strip()]
    if not cleaned_texts:
        return False

    message_content = separator.join(cleaned_texts)
    entry_timestamp_ms = timestamp_s * 1000

    if history_list and history_list[-1].get("role") == role:
        existing_message = history_list[-1]["mensaje"]
        existing_texts_in_last_entry = []
        if isinstance(existing_message, str):
            existing_texts_in_last_entry = existing_message.split(separator)

        merged_texts = [
            t for t in existing_texts_in_last_entry if t] + cleaned_texts

        history_list[-1]["mensaje"] = separator.join(merged_texts)
        history_list[-1]["timestamp_ms"] = max(
            history_list[-1].get("timestamp_ms", 0), entry_timestamp_ms)
        return True
    else:
        history_list.append({
            "role": role,
            "mensaje": message_content,
            "timestamp_ms": entry_timestamp_ms
        })
        return True


async def call_media_endpoint(server: str, session: str, token: str, gemini_api_key: str, telefono: str, gemini_model_name: str) -> Optional[Dict[str, Any]]:
    media_api_url = os.getenv(
        "MEDIA_API_URL", "http://100.42.185.2:8011/get-last-media/")
    headers = {"Content-Type": "application/json"}
    log_prefix = f"[{session}/{telefono}]"
    payload = {
        "sesion": session,
        "token": token,
        "gemini_api_key": gemini_api_key,
        "telefono": telefono,
        "gemini_model_name": gemini_model_name,  # Nuevo parámetro
        "server": server  # Agregar el parámetro server
    }
    # Log para depuración
    print(f"{log_prefix} Calling media endpoint with payload: {payload}")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(media_api_url, json=payload, timeout=60.0)
        response.raise_for_status()
        response_data = response.json()
        return response_data if isinstance(response_data, dict) else {"procesamiento_exitoso": False, "error_detail": "Invalid media response"}
    except httpx.TimeoutException:
        print(f"{log_prefix} Timeout calling media endpoint: {media_api_url}")
        return {"procesamiento_exitoso": False, "error_detail": "Timeout al contactar el servicio de medios"}
    except httpx.HTTPStatusError as e:
        print(
            f"{log_prefix} HTTP error calling media endpoint: {e.response.status_code} - {e.response.text[:200]}")
        return {"procesamiento_exitoso": False, "error_detail": f"Error HTTP {e.response.status_code} del servicio de medios"}
    except Exception as e:
        print(f"{log_prefix} Error calling media endpoint: {e}")
        return {"procesamiento_exitoso": False, "error_detail": str(e)}


async def append_paused_history(req: MessageRequest):
    userbot, phone = req.userbot, req.lineaWA
    log_prefix = f"[{userbot}/{phone}/paused_hist_append]"

    current_state = load_contact_state(userbot, phone)
    history_file = hist_file_path(userbot, phone)
    history_list: List[Dict[str, Any]] = []
    last_timestamp_in_history_ms = 0

    if os.path.exists(history_file):
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history_list = json.load(f)
            if not isinstance(history_list, list):
                history_list = []
            if history_list:
                valid_entries_with_ts = [
                    entry.get("timestamp_ms", 0) for entry in history_list
                    if isinstance(entry, dict) and "timestamp_ms" in entry and isinstance(entry["timestamp_ms"], (int, float))
                ]
                if valid_entries_with_ts:
                    last_timestamp_in_history_ms = max(valid_entries_with_ts)
        except Exception as e:
            print(
                f"{log_prefix} Error loading JSON history {history_file}: {e}. Starting new list.")
            history_list = []

    last_ai_sent_messages_texts: set[str] = set()
    last_ai_turn_timestamp_ms: int = 0
    if history_list:
        for i in range(len(history_list) - 1, -1, -1):
            entry = history_list[i]
            if isinstance(entry, dict) and entry.get("role") == "asistente" and "raw_ai_response_payload" in entry:
                last_ai_turn_timestamp_ms = entry.get("timestamp_ms", 0)
                payload = entry["raw_ai_response_payload"]
                if isinstance(payload, dict):
                    action_payload = {k: v for k,
                                      v in payload.items() if k.isdigit()}
                    sorted_keys = sorted(
                        action_payload.keys(),
                        key=lambda k: (int(k) if isinstance(
                            k, str) and k.isdigit() else float('inf'), k)
                    )
                    for key_val in sorted_keys:
                        action = action_payload.get(key_val)
                        if isinstance(action, dict) and action.get("tipo") == "mensaje":
                            msg_text = action.get("mensaje")
                            if msg_text and isinstance(msg_text, str):
                                last_ai_sent_messages_texts.add(
                                    msg_text.strip())
                break
    AI_ECHO_WINDOW_SECONDS = 15

    timestamps_to_consider = [
        current_state.get("last_control_reaction_timestamp", 0),
        current_state.get("last_message_timestamp_during_pause_ms", 0),
        last_timestamp_in_history_ms
    ]
    timestamps_to_consider = [
        ts for ts in timestamps_to_consider if isinstance(ts, (int, float)) and ts > 0]

    if current_state.get("pause_start_time"):
        try:
            pause_start_dt = datetime.fromisoformat(
                current_state["pause_start_time"])
            pause_start_ts_ms = int(pause_start_dt.timestamp() * 1000)
            if pause_start_ts_ms > 0:
                timestamps_to_consider.append(pause_start_ts_ms)
        except ValueError:
            print(
                f"{log_prefix} Warning: Could not parse pause_start_time '{current_state['pause_start_time']}' for timestamp comparison.")

    since_timestamp_ms = max(
        timestamps_to_consider) if timestamps_to_consider else 0
    new_raw_messages = await fetch_chat_history_since_timestamp_ms(userbot, req.token, phone, since_timestamp_ms, req.server)

    max_ts_from_raw_fetch_ms = since_timestamp_ms
    if new_raw_messages:
        timestamps_in_fetch = [
            int(msg.get("messageTimestamp", 0)) * 1000
            for msg in new_raw_messages if msg.get("messageTimestamp") and str(msg.get("messageTimestamp")).isdigit()
        ]
        if timestamps_in_fetch:
            max_ts_from_raw_fetch_ms = max(
                max_ts_from_raw_fetch_ms, max(timestamps_in_fetch))

    if not new_raw_messages:
        if max_ts_from_raw_fetch_ms > current_state.get("last_message_timestamp_during_pause_ms", 0):
            current_state["last_message_timestamp_during_pause_ms"] = max_ts_from_raw_fetch_ms
            save_contact_state(userbot, phone, current_state)
        return {"status": "paused_no_new_messages", "detail": "No new messages from API for JSON history list."}

    messages_to_process = [
        msg for msg in new_raw_messages
        if int(msg.get("messageTimestamp", 0)) * 1000 > last_timestamp_in_history_ms
    ]

    if not messages_to_process:
        if max_ts_from_raw_fetch_ms > current_state.get("last_message_timestamp_during_pause_ms", 0):
            current_state["last_message_timestamp_during_pause_ms"] = max_ts_from_raw_fetch_ms
            save_contact_state(userbot, phone, current_state)
        return {"status": "paused_no_strictly_newer_messages", "detail": "No messages strictly newer than last history entry found."}

    current_group_sender_from_me: Optional[bool] = None
    current_group_texts: List[str] = []
    latest_ts_in_current_group_s: int = 0
    history_changed_flag = False

    for msg_data in messages_to_process:
        msg_from_me = msg_data.get('key', {}).get('fromMe', False)
        message_obj = msg_data.get('message')

        base_text_from_message = ""
        detected_media_type_for_api: Optional[str] = None

        if isinstance(message_obj, dict):
            if 'conversation' in message_obj and message_obj['conversation']:
                base_text_from_message = message_obj['conversation']
            elif 'extendedTextMessage' in message_obj and isinstance(message_obj['extendedTextMessage'], dict) and message_obj['extendedTextMessage'].get('text'):
                base_text_from_message = message_obj['extendedTextMessage']['text']
            elif 'imageMessage' in message_obj and isinstance(message_obj['imageMessage'], dict) and message_obj['imageMessage'].get('caption'):
                base_text_from_message = message_obj['imageMessage']['caption']
                detected_media_type_for_api = "imagen"
            elif 'videoMessage' in message_obj and isinstance(message_obj['videoMessage'], dict) and message_obj['videoMessage'].get('caption'):
                base_text_from_message = message_obj['videoMessage']['caption']
                detected_media_type_for_api = "video"

            if isinstance(base_text_from_message, str):
                base_text_from_message = base_text_from_message.strip()
            else:
                base_text_from_message = ""

            if not detected_media_type_for_api:
                if 'audioMessage' in message_obj:
                    detected_media_type_for_api = "audio"
                elif 'imageMessage' in message_obj:
                    detected_media_type_for_api = "imagen"
                elif 'videoMessage' in message_obj:
                    detected_media_type_for_api = "video"
                elif 'documentMessage' in message_obj:
                    detected_media_type_for_api = "documento"

            media_info_log_part = ""
            if detected_media_type_for_api and not msg_from_me:
                print(
                    f"{log_prefix} Paused msg: User sent '{detected_media_type_for_api}'. Processing with media endpoint.")
                media_response = await call_media_endpoint(
                    req.server, req.userbot, req.token, req.apikey, phone, req.ai_model  # Pasar ai_model
                )

                if media_response and media_response.get("procesamiento_exitoso"):
                    processed_text = media_response.get("procesamiento_gemini")
                    if processed_text and isinstance(processed_text, str) and processed_text.strip():
                        transcription = processed_text.strip()
                        media_info_log_part = f" [Transcripción/descripción {detected_media_type_for_api}: '{transcription}']"
                    else:
                        no_text_msg = "No se obtuvo texto o el texto está vacío."
                        media_info_log_part = f" [Error procesando {detected_media_type_for_api}: {no_text_msg}]"
                else:
                    error_detail = media_response.get(
                        'error_detail', 'Desconocido') if media_response else 'Fallo endpoint de medios'
                    media_info_log_part = f" [Error procesando {detected_media_type_for_api}: {error_detail}]"

            if media_info_log_part:
                if base_text_from_message:
                    text_content = base_text_from_message + media_info_log_part
                else:
                    text_content = media_info_log_part.strip()
            else:
                text_content = base_text_from_message
        else:
            text_content = ""

        msg_timestamp_s_raw = msg_data.get("messageTimestamp", 0)
        msg_timestamp_s = int(msg_timestamp_s_raw) if str(
            msg_timestamp_s_raw).isdigit() else 0
        if msg_timestamp_s == 0:
            continue

        current_message_ts_ms = msg_timestamp_s * 1000
        if msg_from_me and text_content:
            is_self_echo = False
            if last_ai_turn_timestamp_ms > 0 and \
               (current_message_ts_ms >= last_ai_turn_timestamp_ms) and \
               (current_message_ts_ms - last_ai_turn_timestamp_ms) < (AI_ECHO_WINDOW_SECONDS * 1000):
                if text_content.strip() in last_ai_sent_messages_texts:
                    is_self_echo = True
            if is_self_echo:
                continue

        if not text_content:
            if current_group_texts:
                role_to_flush = "operador" if current_group_sender_from_me else "usuario_pausado"
                if _add_or_merge_message_group(history_list, role_to_flush, current_group_texts, latest_ts_in_current_group_s, log_prefix):
                    history_changed_flag = True
                current_group_texts = []
                current_group_sender_from_me = None
            continue

        if current_group_sender_from_me is None or current_group_sender_from_me != msg_from_me:
            if current_group_texts:
                role_to_flush = "operador" if current_group_sender_from_me else "usuario_pausado"
                if _add_or_merge_message_group(history_list, role_to_flush, current_group_texts, latest_ts_in_current_group_s, log_prefix):
                    history_changed_flag = True
            current_group_sender_from_me = msg_from_me
            current_group_texts = [text_content]
            latest_ts_in_current_group_s = msg_timestamp_s
        else:
            current_group_texts.append(text_content)
            latest_ts_in_current_group_s = max(
                latest_ts_in_current_group_s, msg_timestamp_s)

    if current_group_texts:
        role_to_flush = "operador" if current_group_sender_from_me else "usuario_pausado"
        if _add_or_merge_message_group(history_list, role_to_flush, current_group_texts, latest_ts_in_current_group_s, log_prefix):
            history_changed_flag = True

    if max_ts_from_raw_fetch_ms > current_state.get("last_message_timestamp_during_pause_ms", 0):
        current_state["last_message_timestamp_during_pause_ms"] = max_ts_from_raw_fetch_ms
        save_contact_state(userbot, phone, current_state)

    if not history_changed_flag:
        return {"status": "paused_no_new_loggable_text_groups", "detail": "No new loggable text message groups formed/merged for JSON history."}

    try:
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history_list, f, ensure_ascii=False, indent=2)
        print(
            f"{log_prefix} Paused history saved to {history_file}. Changes: {history_changed_flag}")
        return {"status": "paused_history_updated", "detail": "Message group changes/appends to JSON list completed."}
    except Exception as e:
        print(f"{log_prefix} Error saving updated JSON history list: {e}")
        return {"status": "error_saving_paused_json_history", "detail": str(e)}

# Función auxiliar para la llamada síncrona a Gemini SDK


def _call_gemini_sdk_sync(
    api_key_from_req: str,
    user_content: str,
    system_instruction_text: str,
    model_to_use: str,
    temperature_val: float,
    top_p_val: float,
    max_tokens_val: int,
    log_prefix_outer: str,  # Pasar el log_prefix
    pais: str,
    zona_horaria: str,
    fecha_hora_formateada: str,
    thinking_budget: int = 0,  # Nuevo parámetro para thinking_config
    thinking_level: str = "HIGH",  # Nuevo parámetro para Gemini 3.0
    # Nuevo parámetro para Gemini 3.0
    media_resolution: str = "MEDIA_RESOLUTION_HIGH",
    user_push_name: str = "",  # Nuevo parámetro para el pushName del usuario
    use_google_search: bool = False,
    use_google_maps: bool = False
):
    # Esta función ahora es puramente síncrona y no contiene lógica de reintentos ni asyncio.
    # Los errores se propagan hacia arriba.
    print(f"{log_prefix_outer} SDK Call: Model={model_to_use}, Temp={temperature_val}, TopP={top_p_val}, MaxT={max_tokens_val}, ThinkingBudget={thinking_budget}, ThinkingLevel={thinking_level}, MediaRes={media_resolution}, UserPushName='{user_push_name}'")

    sdk_client = genai.Client(api_key=api_key_from_req, http_options={
                              'api_version': 'v1beta'})

    # Construir la parte del nombre del usuario si está disponible
    user_name_part = ""
    if user_push_name:
        user_name_part = f" ({user_push_name})"

    # Construir el contenido formateado para enviar a Gemini
    formatted_user_content = f"""Mensaje ACTUAL del cliente{user_name_part} (puede incluir texto de varios mensajes cortos concatenados, y/o la descripción de un archivo multimedia procesado): {user_content} -> [Fecha y hora actual - {fecha_hora_formateada}]"""

    sdk_contents = [
        genai_types.Content(role="user", parts=[
                            genai_types.Part.from_text(text=formatted_user_content)])
    ]
    sdk_safety_settings = [
        genai_types.SafetySetting(category=genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                                  threshold=genai_types.HarmBlockThreshold.BLOCK_NONE),
        genai_types.SafetySetting(category=genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                  threshold=genai_types.HarmBlockThreshold.BLOCK_NONE),
        genai_types.SafetySetting(category=genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                  threshold=genai_types.HarmBlockThreshold.BLOCK_NONE),
        genai_types.SafetySetting(category=genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                  threshold=genai_types.HarmBlockThreshold.BLOCK_NONE),
    ]

    # Crear la configuración base
    final_sdk_config = genai_types.GenerateContentConfig(
        temperature=temperature_val,
        top_p=top_p_val,
        max_output_tokens=max_tokens_val,
        response_mime_type="application/json",
        safety_settings=sdk_safety_settings,
        system_instruction=[genai_types.Part.from_text(
            text=system_instruction_text)]
    )

    sdk_tools = []
    if use_google_search:
        print(f"{log_prefix_outer} [TOOLS] Habilitando Búsqueda en Google")
        sdk_tools.append(genai_types.Tool(
            google_search=genai_types.GoogleSearch()))
        if use_google_maps:
            print(
                f"{log_prefix_outer} [TOOLS-WARN] Se solicitaron Google Search y Google Maps. La API no permite combinarlos. Se priorizará Google Search.")
    elif use_google_maps:
        print(f"{log_prefix_outer} [TOOLS] Habilitando Google Maps")
        sdk_tools.append(genai_types.Tool(
            google_maps=genai_types.GoogleMaps()))
    if sdk_tools:
        final_sdk_config.tools = sdk_tools

    # Agregar media_resolution
    if "gemini-3" in model_to_use:
        try:
            # En la API de Google GenAI el valor correcto para resolución es MEDIA_RESOLUTION_HIGH
            # Para evitar el UserWarning de Pydantic por tipo de dato, asignamos el valor del Enum directamente si existe
            media_resolution_to_apply = media_resolution if isinstance(
                media_resolution, str) else "MEDIA_RESOLUTION_HIGH"

            print(
                f"{log_prefix_outer} Aplicando media_resolution: {media_resolution_to_apply}")

            # Intento de parsear al Enum si el SDK lo provee (google.genai)
            if hasattr(genai_types, 'MediaResolution'):
                # Convertimos string a Enum si es posible
                try:
                    final_sdk_config.media_resolution = genai_types.MediaResolution[
                        media_resolution_to_apply]
                except KeyError:
                    # Si falla, mandamos string y asumimos el warning
                    final_sdk_config.media_resolution = media_resolution_to_apply
            else:
                final_sdk_config.media_resolution = media_resolution_to_apply

        except Exception as e:
            print(
                f"{log_prefix_outer} Warn: media_resolution no soportado en SDK actual: {e}")

    # Agregar thinking_config solo para modelos compatibles
    if model_to_use in THINKING_COMPATIBLE_MODELS:
        # Para Gemini 3.0+ usamos SOLO thinking_level y NO enviamos thinking_budget NUNCA
        if "gemini-3" in model_to_use:
            # -1 en budget desde versiones anteriores significa habilitado por defecto, o usamos el nivel explícito
            level_to_use = thinking_level if thinking_level else "HIGH"

            print(f"{log_prefix_outer} Aplicando thinking_config con level {level_to_use} para modelo compatible: {model_to_use}")
            # ATENCIÓN: Pydantic falla si se envía `thinking_budget` en modelos 3.0.
            # Por eso SOLO enviamos thinking_level en kwargs explícitos para no enviar defaults accidentales
            final_sdk_config.thinking_config = genai_types.ThinkingConfig(
                thinking_level=level_to_use
            )
        else:
            # Modelos 2.0 / 2.5 usan thinking_budget numérico
            if thinking_budget and thinking_budget != 0:
                print(
                    f"{log_prefix_outer} Aplicando thinking_config con budget {thinking_budget} para modelo compatible: {model_to_use}")
                final_sdk_config.thinking_config = genai_types.ThinkingConfig(
                    thinking_budget=thinking_budget
                )

    stream_text_parts = []
    response_stream_obj = None

    for chunk_response in sdk_client.models.generate_content_stream(
        model=model_to_use,
        contents=sdk_contents,
        config=final_sdk_config
    ):
        response_stream_obj = chunk_response
        if chunk_response.text:
            stream_text_parts.append(chunk_response.text)

    if not response_stream_obj:
        raise Exception(
            f"[{log_prefix_outer}] Gemini SDK stream returned no response object / was empty for model {model_to_use}.")

    final_text_output_from_api = "".join(stream_text_parts)

    if hasattr(response_stream_obj, 'candidates') and response_stream_obj.candidates:
        candidate = response_stream_obj.candidates[0]
        if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
            print(
                f"{log_prefix_outer} [TOOLS-SUCCESS] Metadata de Grounding detectada! El modelo buscó en Google u obtuvo datos externos.")
            # Opcionalmente imprimir los chunks usados:
            if hasattr(candidate.grounding_metadata, 'grounding_chunks'):
                print(
                    f"{log_prefix_outer} [TOOLS-CHUNKS] Se usaron {len(candidate.grounding_metadata.grounding_chunks)} fuentes.")
        else:
            if sdk_tools:
                print(
                    f"{log_prefix_outer} [TOOLS-INFO] Las herramientas estaban habilitadas, pero el modelo NO hizo uso de ellas para esta consulta.")

    if response_stream_obj.prompt_feedback and response_stream_obj.prompt_feedback.block_reason:
        block_reason_name_str = response_stream_obj.prompt_feedback.block_reason.name
        raise GeminiBlockedError(
            f"[{log_prefix_outer}] Gemini request blocked (prompt_feedback) for model {model_to_use} due to: {block_reason_name_str}", block_reason_name_str)

    if response_stream_obj.candidates:
        for candidate in response_stream_obj.candidates:
            if candidate.finish_reason.name not in ['STOP', 'MAX_TOKENS', 'UNSPECIFIED']:
                if candidate.finish_reason.name == 'SAFETY':
                    block_reason_detail = f"candidate_safety_block_model_{model_to_use}"
                    if candidate.safety_ratings:
                        safety_ratings_str = ", ".join(
                            [f"{sr.category.name}: {sr.probability.name}" for sr in candidate.safety_ratings])
                        block_reason_detail = f"candidate_safety_ratings_model_{model_to_use} ({safety_ratings_str})"
                    raise GeminiBlockedError(
                        f"[{log_prefix_outer}] Gemini candidate generation stopped for model {model_to_use} due to safety: {candidate.finish_reason.name}", block_reason_detail)
                else:
                    raise Exception(
                        f"[{log_prefix_outer}] Gemini candidate generation for model {model_to_use} stopped unexpectedly: {candidate.finish_reason.name}")

    if not final_text_output_from_api:  # No explicit block, but empty text
        # Esto podría ser un JSON vacío "{}" si responseMimeType es JSON, lo cual es válido pero no útil aquí.
        # O un error no capturado.
        # La validación de JSON posterior lo detectará si es inválido.
        # Si es un string vacío "" y se esperaba JSON, json.loads fallará.
        print(f"[{log_prefix_outer}] Warning: Gemini SDK for model {model_to_use} returned empty text output. Further JSON parsing will determine validity.")

    return final_text_output_from_api


async def process_message_final(req: MessageRequest, message_fragments: List[str], pais: str, idioma: str, user_push_name: str = '') -> Optional[Dict[str, Any]]:
    # Extraer los datos y limpiarlos de espacios en blanco
    phone = req.lineaWA.strip() if req.lineaWA else None
    userbot = req.userbot.strip() if req.userbot else None
    if getattr(req, "activaruserbotopcional", False) and getattr(req, "userbotopcional", None):
        userbot = req.userbotopcional.strip()

    log_prefix = f"[{userbot}/{phone}]"

    base_text_from_fragments = ", ".join(message_fragments).strip()

    history_file = hist_file_path(userbot, phone)
    history_list_from_file: List[Dict[str, Any]] = []
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history_list_from_file = json.load(f)
            if not isinstance(history_list_from_file, list):
                history_list_from_file = []
        except Exception as e:
            print(f"{log_prefix} Error loading JSON history {history_file}: {e}")
            history_list_from_file = []

    client_message_for_history = base_text_from_fragments
    final_content_for_ai = base_text_from_fragments

    media_emoji_map = {'🎤': 'audio', '📷': 'imagen',
                       '📄': 'documento', '📹': 'video'}
    detected_media_type = None
    media_url_from_endpoint = None

    for fragment in message_fragments:
        for emoji, media_type_key in media_emoji_map.items():
            if emoji in fragment:
                detected_media_type = media_type_key
                break
        if detected_media_type:
            break

    if detected_media_type:
        print(
            f"{log_prefix} Media emoji '{detected_media_type}' detected. Calling media endpoint.")
        media_response = await call_media_endpoint(
            # Pasar ai_model de la request
            req.server, req.userbot, req.token, req.apikey, phone, req.ai_model
        )

        media_descriptor_for_prompt = f"\n[Adjunto (Tipo: {detected_media_type}): "
        media_transcription_for_history_log = ""

        if media_response and media_response.get("procesamiento_exitoso"):
            processed_text = media_response.get("procesamiento_gemini")
            media_url_from_endpoint = media_response.get("url")

            if processed_text and isinstance(processed_text, str) and processed_text.strip():
                transcription = processed_text.strip()
                if detected_media_type == "audio":
                    media_descriptor_for_prompt += f"Transcripcion de audio que envio el cliente: '{transcription}'"
                else:
                    media_descriptor_for_prompt += f"Contenido/descripcion detectada: '{transcription}'"
                media_transcription_for_history_log = f" [Transcripción/descripción {detected_media_type}: '{transcription}']"
            else:
                no_text_msg = "No se obtuvo texto o el texto está vacío."
                media_descriptor_for_prompt += no_text_msg
                media_transcription_for_history_log = f" [Error procesando {detected_media_type}: {no_text_msg}]"

            if detected_media_type in ["imagen", "video", "documento"] and media_url_from_endpoint:
                media_descriptor_for_prompt += f" URL: {media_url_from_endpoint}"
        else:
            error_detail = media_response.get(
                'error_detail', 'Desconocido') if media_response else 'Fallo endpoint de medios'
            media_descriptor_for_prompt += f"Error al procesar adjunto: {error_detail}"
            media_transcription_for_history_log = f" [Error procesando {detected_media_type}: {error_detail}]"

        media_descriptor_for_prompt += "]"
        client_message_for_history += media_transcription_for_history_log
        final_content_for_ai += media_descriptor_for_prompt

    plain_history_for_prompt = ""
    num_messages_to_take = req.numerodemensajes
    relevant_history_items = []
    if num_messages_to_take > 0:
        start_index = max(0, len(history_list_from_file) -
                          num_messages_to_take)
        relevant_history_items = history_list_from_file[start_index:]

    num_actual_items_shown = len(relevant_history_items)

    items_for_prompt_history = relevant_history_items
    if relevant_history_items and num_actual_items_shown > 0:
        last_item_from_loaded_history = relevant_history_items[-1]
        if last_item_from_loaded_history.get("role") in ["cliente", "usuario_pausado"]:
            separator_hist = ", "
            fragments_from_last_history_item = [
                text.strip() for text in last_item_from_loaded_history.get("mensaje", "").strip().split(separator_hist) if text.strip()
            ]
            current_fragments_for_prompt_check = [
                f.strip() for f in base_text_from_fragments.split(separator_hist) if f.strip()]

            if fragments_from_last_history_item == current_fragments_for_prompt_check and fragments_from_last_history_item:
                items_for_prompt_history = relevant_history_items[:-1]
                num_actual_items_shown = len(items_for_prompt_history)

    formatted_history_blocks = []
    if items_for_prompt_history:
        for item in items_for_prompt_history:
            role = item.get("role")
            message_text = item.get("mensaje", "").strip()
            item_media_url = item.get("media_url")
            item_media_type = item.get("media_type")
            item_estado_conv = item.get("estado_conversacion")
            item_timestamp = item.get("timestamp_ms")

            if not message_text:
                continue
            block_text = "=============================\n"
            if role in ["cliente", "usuario_pausado"]:
                block_text += f"🗣️ Cliente:\n{message_text}\n"
                if item_media_url and item_media_type in ["imagen", "video", "documento"]:
                    block_text += f"(URL del medio: {item_media_url})\n"
            elif role in ["asistente", "operador"]:
                block_text += f"🤖 Asistente IA/Operador:\n{message_text}\n"
                if item_estado_conv and role == "asistente":
                    block_text += f"(Estado interno IA previo: {item_estado_conv})\n"
            else:
                continue

            # Agregar timestamp legible al final del mensaje
            # Siempre intentar agregar fecha, usando timestamp actual si no existe
            timestamp_to_use = item_timestamp
            if not timestamp_to_use or timestamp_to_use == 0:
                # Si no hay timestamp, usar timestamp actual
                timestamp_to_use = int(time.time() * 1000)
                print(
                    f"[DEBUG] No timestamp found for message, using current time: {timestamp_to_use}")

            try:
                print(
                    f"[DEBUG] Processing timestamp: {timestamp_to_use} for country: {pais}")
                readable_timestamp = format_timestamp_with_timezone(
                    timestamp_to_use, pais)
                print(f"[DEBUG] Formatted timestamp: {readable_timestamp}")
                block_text += f"📅 {readable_timestamp}\n"
            except Exception as e:
                print(
                    f"[ERROR] Failed to format timestamp {timestamp_to_use}: {e}")
                # Fallback: mostrar fecha actual en formato simple
                try:
                    timezone_str = get_timezone_from_country(pais)
                    tz = pytz.timezone(timezone_str)
                    now_local = datetime.now(tz)
                    fallback_date = now_local.strftime('%d/%m/%Y %I:%M:%S %p')
                    block_text += f"📅 {fallback_date} ({timezone_str})\n"
                except:
                    # Último fallback: UTC
                    utc_now = datetime.now(pytz.utc)
                    fallback_date = utc_now.strftime('%d/%m/%Y %I:%M:%S %p')
                    block_text += f"📅 {fallback_date} (UTC)\n"

            formatted_history_blocks.append(block_text)
        if formatted_history_blocks:
            plain_history_for_prompt = "".join(formatted_history_blocks)
            plain_history_for_prompt += "=============================\n\n"
        else:
            plain_history_for_prompt = "(No hay historial relevante para la IA)\n"
    else:
        plain_history_for_prompt = "(No se incluyó historial para la IA)\n"

    timezone_str = get_timezone_from_country(pais)
    try:
        tz = pytz.timezone(timezone_str)
    except pytz.UnknownTimeZoneError:
        tz = pytz.utc
        timezone_str = "UTC"
    now_local = datetime.now(tz)
    try:
        loc_day = babel.dates.format_datetime(
            now_local, format='EEEE', locale=idioma).capitalize()
    except Exception:
        loc_day = now_local.strftime('%A')
    try:
        loc_date = babel.dates.format_date(
            now_local, format='long', locale=idioma)
    except Exception:
        loc_date = now_local.strftime('%d/%m/%Y')
    try:
        loc_time_val = now_local.strftime('%I:%M %p')
    except Exception:
        loc_time_val = now_local.strftime('%H:%M')
    loc_time = loc_time_val.replace('am', 'AM').replace(
        'pm', 'PM') + f" ({timezone_str})"
    formatted_dt_prompt = f"Es el dia {loc_day} {loc_date} y son las {loc_time}"

    prompt_data = {
        "system_rules_base": req.promt,
        "num_mensajes_shown": num_actual_items_shown,
        "historial_texto": plain_history_for_prompt,
        "idioma_respuesta": idioma,
        "example_json_structure_placeholder": EXAMPLE_JSON_STRUCTURE
    }

    # Generar prompt dinámicamente basado en los parámetros de estados
    dynamic_prompt_template = get_prompt_template(
        estados_generales=req.estados_generales,
        estados_especificos=req.estados_especificos
    )
    current_prompt_text_for_system_instruction = dynamic_prompt_template.format(
        **prompt_data)

    # LOG para mostrar el prompt final y el contenido que se enviará a la IA
    # print(f"{log_prefix} ===== PROMPT Y CONTENIDO ENVIADO A LA IA =====")
    # print(f"{log_prefix} USER_CONTENT (final_content_for_ai):")
    # print(f"{log_prefix} {final_content_for_ai}")
    # print(f"{log_prefix} " + "="*50)
    # print(f"{log_prefix} SYSTEM_INSTRUCTION (prompt final):")
    # print(f"{log_prefix} {current_prompt_text_for_system_instruction}")
    # print(f"{log_prefix} " + "="*50)
    print(f"{log_prefix} PAIS: {pais}")
    print(f"{log_prefix} ZONA_HORARIA: {timezone_str}")
    print(f"{log_prefix} FECHA_HORA_FORMATEADA: {formatted_dt_prompt}")

    # Logs detallados de parámetros por modelo
    if req.ai_model and "gemini-3" in req.ai_model:
        print(f"{log_prefix} PARAMETROS IA (GEMINI 3.0+): Level={getattr(req, 'thinking_level', 'HIGH')}, MediaRes={getattr(req, 'media_resolution', 'MEDIA_RESOLUTION_HIGH')}")
    else:
        print(f"{log_prefix} PARAMETROS IA (GEMINI <3.0): Budget={req.thinking_budget}")

    print(f"{log_prefix} " + "="*50)

    # === VERIFICAR LANGGRAPH (Bot Activo en DB) ===
    use_langgraph = False
    bot_config = None
    try:
        from core.database import SessionLocal
        from models import bot_models
        db = SessionLocal()
        bot_config = db.query(bot_models.BotConfig).filter(
            bot_models.BotConfig.userbot_identifier == userbot).first()
        if bot_config and bot_config.is_active:
            use_langgraph = True

            try:
                tools_cfg = json.loads(bot_config.tools_config or "{}")
            except:
                tools_cfg = {}

            config_dict = {
                "api_key": bot_config.apikey or req.apikey,
                "ai_model": req.ai_model or bot_config.ai_model,
                "tools_config": tools_cfg,
                "bot_id": bot_config.id,
                "userbot_identifier": bot_config.userbot_identifier,
                "use_google_search": getattr(req, 'use_google_search', False) or bot_config.use_google_search,
                "use_google_maps": getattr(req, 'use_google_maps', False) or bot_config.use_google_maps,
                "fallback_models": GEMINI_FALLBACK_MODELS_LIST
            }
        db.close()
    except Exception as e:
        print(f"{log_prefix} ❌ Error verificando BD para LangGraph: {e}")

    if use_langgraph:
        from services.graph import process_message_with_graph
        print(
            f"{log_prefix} 🧠 [LangGraph] Procesando mensaje con LangGraph...")
        try:
            loop = asyncio.get_event_loop()
            raw_text_from_gemini = await loop.run_in_executor(
                None,
                process_message_with_graph,
                bot_config.id,
                phone,
                final_content_for_ai,
                current_prompt_text_for_system_instruction,
                config_dict,
                history_list_from_file,
                user_push_name,
                formatted_dt_prompt
            )

            try:
                ai_response_json_payload = json.loads(raw_text_from_gemini)
                print(f"{log_prefix} ✅ [LangGraph] Parseo exitoso del JSON.")
            except json.JSONDecodeError as e:
                print(
                    f"{log_prefix} ❌ Error decodificando JSON de LangGraph: {e}. Payload: {raw_text_from_gemini}")
                # Intentar extraer JSON de la respuesta bruta si hay markdown json
                import re
                json_match = re.search(
                    r'```(?:json)?(.*?)```', raw_text_from_gemini, re.DOTALL)
                if json_match:
                    try:
                        ai_response_json_payload = json.loads(
                            json_match.group(1).strip())
                        print(
                            f"{log_prefix} ✅ [LangGraph] Parseo exitoso tras extraer bloque markdown.")
                    except Exception as ex:
                        pass

                if ai_response_json_payload is None:
                    ai_response_json_payload = {"1": {
                        "tipo": "mensaje", "mensaje": raw_text_from_gemini}, "estado_conversacion": "procesando"}

        except Exception as e:
            print(
                f"{log_prefix} ❌ Error en LangGraph: {e}. Haremos fallback a legacy sin tools.")

    last_overall_error_type = "initial_no_model_attempted"

    # Si no se obtuvo payload de LangGraph, usar fallback
    # Construir lista de modelos a intentar: el de la request primero, luego los de fallback sin duplicados.
    model_names_to_try_ordered = []
    seen_models = set()
    if req.ai_model and req.ai_model.strip():
        model_names_to_try_ordered.append(req.ai_model.strip())
        seen_models.add(req.ai_model.strip())

    for fallback_model in GEMINI_FALLBACK_MODELS_LIST:
        if fallback_model not in seen_models:
            model_names_to_try_ordered.append(fallback_model)
            seen_models.add(fallback_model)

    if "gemini-2.0-flash" not in model_names_to_try_ordered:
        print(f"{log_prefix} Añadiendo gemini-2.0-flash como fallback seguro.")
        model_names_to_try_ordered.append("gemini-2.0-flash")

    # Log mejorado para mostrar TODOS los modelos que se van a intentar
    print(f"{log_prefix} ===== MODELOS A INTENTAR =====")
    print(f"{log_prefix} Modelo principal (de la request): {req.ai_model}")
    print(f"{log_prefix} Lista completa de modelos a intentar en orden: {model_names_to_try_ordered}")
    print(f"{log_prefix} Total de modelos a intentar: {len(model_names_to_try_ordered)}")
    print(f"{log_prefix} Modelos compatibles con thinking_config: {THINKING_COMPATIBLE_MODELS}")
    print(f"{log_prefix} ==============================")

    if ai_response_json_payload is None:
        for model_index, current_model_sdk in enumerate(model_names_to_try_ordered):
            print(
                f"{log_prefix} [MODELO {model_index + 1}/{len(model_names_to_try_ordered)}] Intentando con modelo Gemini: {current_model_sdk}")
            # Número de reintentos para errores transitorios (como sobrecarga) para ESTE modelo
            # 1 intento original + 1 reintento = 2 intentos totales por modelo para sobrecarga
            max_retries_per_model = 1

            for attempt in range(max_retries_per_model + 1):  # Intentos: 0, 1
                try:
                    loop = asyncio.get_event_loop()
                    raw_text_from_gemini = await loop.run_in_executor(
                        None,
                        _call_gemini_sdk_sync,  # Función síncrona auxiliar
                        req.apikey,
                        final_content_for_ai,
                        current_prompt_text_for_system_instruction,
                        current_model_sdk,
                        req.temperature,
                        req.topP,
                        65536,  # Forzado a 65536 tokens independientemente de req.maxOutputTokens
                        log_prefix,  # Pasar log_prefix para logging interno en _call_gemini_sdk_sync
                        pais,
                        timezone_str,
                        formatted_dt_prompt,
                        # Evitar inyección de budget en modelos 3.0 desde el nivel superior
                        0 if "gemini-3" in current_model_sdk else getattr(
                            req, 'thinking_budget', -1),
                        getattr(req, 'thinking_level', 'HIGH'),
                        getattr(req, 'media_resolution',
                                'MEDIA_RESOLUTION_HIGH'),
                        user_push_name,  # Pasar el pushName a _call_gemini_sdk_sync
                        getattr(req, 'use_google_search', False),
                        getattr(req, 'use_google_maps', False)
                    )

                    print(
                        f"{log_prefix} [RAW OUTPUT FROM SDK - MODEL: {current_model_sdk}]:\n{raw_text_from_gemini}\n")

                    try:
                        # Limpieza básica para evitar errores comunes de JSON (espacios, saltos de línea al inicio/fin, o bloques de código markdown)
                        cleaned_raw_text = raw_text_from_gemini.strip()
                        if cleaned_raw_text.startswith("```json"):
                            cleaned_raw_text = cleaned_raw_text[7:]
                        if cleaned_raw_text.startswith("```"):
                            cleaned_raw_text = cleaned_raw_text[3:]
                        if cleaned_raw_text.endswith("```"):
                            cleaned_raw_text = cleaned_raw_text[:-3]
                        cleaned_raw_text = cleaned_raw_text.strip()

                        parsed_json = json.loads(cleaned_raw_text)
                        if isinstance(parsed_json, dict) and "estado_conversacion" in parsed_json and isinstance(parsed_json["estado_conversacion"], str):
                            action_items_valid = True
                            has_action_items = False
                            for k, v_action in parsed_json.items():
                                if k.isdigit():
                                    has_action_items = True
                                    if not (isinstance(v_action, dict) and 'tipo' in v_action):
                                        action_items_valid = False
                                        break
                                elif k == "estado_conversacion":
                                    continue

                            if action_items_valid and has_action_items:
                                ai_response_json_payload = parsed_json
                                last_overall_error_type = ""  # Éxito
                                print(
                                    f"{log_prefix} ✅ ÉXITO con modelo {current_model_sdk} (Modelo {model_index + 1}/{len(model_names_to_try_ordered)}).")
                                # Salir del bucle de reintentos (attempt)
                                break
                            else:
                                # Error de estructura, pero la llamada a la API fue "exitosa" (no lanzó excepción)
                                last_overall_error_type = f"json_actions_invalid_model_{current_model_sdk}"
                                error_detail_str = f"Estructura de acciones JSON inválida o sin acciones (modelo {current_model_sdk}): {str(parsed_json)[:200]}"
                                print(f"{log_prefix} ❌ {error_detail_str}")
                                ai_response_json_payload = None
                                # No reintentar este modelo por este error de formato, pasar al siguiente modelo.
                                break  # Salir del bucle de reintentos para este modelo
                        else:
                            last_overall_error_type = f"json_missing_estado_conv_model_{current_model_sdk}"
                            error_detail_str = f"Falta 'estado_conversacion' o tipo incorrecto (modelo {current_model_sdk}): {str(parsed_json)[:200]}"
                            print(f"{log_prefix} ❌ {error_detail_str}")
                            ai_response_json_payload = None
                            # No reintentar este modelo por este error de formato, pasar al siguiente.
                            break  # Salir del bucle de reintentos para este modelo
                    except json.JSONDecodeError as json_e:
                        last_overall_error_type = f"json_decode_error_model_{current_model_sdk}"
                        print(
                            f"{log_prefix} ❌ Gemini JSON decode error (modelo {current_model_sdk}): {json_e}. Intentando reparar...")

                        # Intento de reparación de JSON incompleto (muy común en respuestas cortadas)
                        try:
                            import re
                            # Intentar cerrar strings, diccionarios y comillas simples que falten al final
                            repaired_text = cleaned_raw_text
                            if repaired_text.count('"') % 2 != 0:
                                repaired_text += '"'
                            if not repaired_text.endswith("}"):
                                # A veces se corta a la mitad de un objeto anidado, intentar cerrar lo básico
                                repaired_text += "}}" if repaired_text.endswith(
                                    '"') else '"}}'

                            # Limpiar trailing comas antes de cerrar llaves
                            repaired_text = re.sub(
                                r',\s*}', '}', repaired_text)

                            parsed_json = json.loads(repaired_text)

                            # Si se logra parsear tras reparar, validar estructura mínima
                            if isinstance(parsed_json, dict) and "estado_conversacion" in parsed_json:
                                print(
                                    f"{log_prefix} ⚠️ JSON reparado exitosamente para modelo {current_model_sdk}.")
                                ai_response_json_payload = parsed_json
                                last_overall_error_type = ""
                                break
                            else:
                                raise Exception(
                                    "JSON reparado pero estructura no válida.")
                        except Exception as repair_e:
                            print(
                                f"{log_prefix} ❌ Falló reparación de JSON (modelo {current_model_sdk}): {repair_e}. Raw original: {raw_text_from_gemini[:200]}")
                            ai_response_json_payload = None

                            # Si es el primer intento, vamos a reintentar la llamada al SDK porque a veces Gemini se traba
                            if attempt < max_retries_per_model:
                                print(
                                    f"{log_prefix} ⏳ Reintentando modelo {current_model_sdk} tras fallo de JSON...")
                                await asyncio.sleep(2)
                                continue  # Forzar nuevo intento al SDK
                            else:
                                break  # Salir al siguiente modelo

                except GeminiBlockedError as e_block:
                    last_overall_error_type = f"gemini_blocked_{e_block.block_reason_name}_model_{current_model_sdk}"
                    print(
                        f"{log_prefix} ⛔ {str(e_block)}. No se intentarán más modelos para este error de bloqueo.")
                    ai_response_json_payload = None
                    # Este error es final, salir del bucle de modelos también
                    # Para salir del bucle de modelos, podemos re-lanzar o tener una bandera
                    # Re-lanzar para que el except exterior del bucle de modelos lo capture.
                    raise

                except Exception as e_sdk:  # Captura errores de _call_gemini_sdk_sync
                    error_msg_lower = str(e_sdk).lower()
                    # Ajustar estas detecciones según los mensajes de error reales de la SDK de genai
                    is_overloaded_error = "overloaded" in error_msg_lower or \
                        "503" in error_msg_lower or \
                        "unavailable" in error_msg_lower or \
                        "try again later" in error_msg_lower

                    is_invalid_model_error = ("invalid" in error_msg_lower and ("model name" in error_msg_lower or "argument" in error_msg_lower)) or \
                                             ("not found" in error_msg_lower and "model" in error_msg_lower) or \
                                             ("permission denied" in error_msg_lower and f"model '{current_model_sdk}'" in error_msg_lower) or \
                                             (f"model '{current_model_sdk}' not found" in error_msg_lower)

                    last_overall_error_type = f"sdk_error_model_{current_model_sdk}_attempt_{attempt+1}: {str(e_sdk)[:150]}"
                    print(
                        f"{log_prefix} ⚠️ Error SDK con modelo {current_model_sdk} (intento {attempt+1}/{max_retries_per_model+1}): {e_sdk}")

                    if is_invalid_model_error:
                        print(
                            f"{log_prefix} ❌ Modelo {current_model_sdk} inválido o no accesible. Pasando al siguiente modelo.")
                        ai_response_json_payload = None
                        # Salir del bucle de reintentos (attempt), para probar el siguiente modelo en el bucle exterior.
                        break

                    if is_overloaded_error:
                        if attempt < max_retries_per_model:
                            sleep_duration = (attempt + 2) * 3
                            print(
                                f"{log_prefix} ⏳ Modelo {current_model_sdk} sobrecargado. Reintentando en {sleep_duration}s...")
                            await asyncio.sleep(sleep_duration)
                            # continue al siguiente intento para ESTE modelo
                        else:
                            print(
                                f"{log_prefix} ❌ Modelo {current_model_sdk} sigue sobrecargado tras {max_retries_per_model+1} intentos. Pasando al siguiente modelo.")
                            ai_response_json_payload = None
                            # Salir del bucle de reintentos (attempt), para probar el siguiente modelo.
                            break

                    # Otros errores de SDK (ej. red, timeout dentro de la SDK si no es capturado antes)
                    # Si no es sobrecarga ni modelo inválido, y no es el último intento para este modelo:
                    elif attempt < max_retries_per_model:
                        sleep_duration = (attempt + 1) * 2
                        print(
                            f"{log_prefix} ⚠️ Error SDK general. Reintentando modelo {current_model_sdk} en {sleep_duration}s: {last_overall_error_type}")
                        await asyncio.sleep(sleep_duration)
                        # continue al siguiente intento para ESTE modelo
                    else:  # Último intento para este modelo falló con error general
                        print(
                            f"{log_prefix} ❌ Error SDK general persistente con modelo {current_model_sdk} tras {max_retries_per_model+1} intentos. Pasando al siguiente modelo.")
                        ai_response_json_payload = None
                        # Salir del bucle de reintentos (attempt), para probar el siguiente modelo.
                        break

                # Si después del bucle de reintentos (attempt) se obtuvo un payload, salir del bucle de modelos
                if ai_response_json_payload is not None:
                    break

            # Si el payload sigue siendo None después de los reintentos para el modelo actual,
            # el bucle de modelos (current_model_sdk) continuará con el siguiente modelo.
                if ai_response_json_payload is not None:
                    # Salir del bucle de modelos (current_model_sdk) porque tuvimos éxito
                    break
                else:
                    print(
                        f"{log_prefix} ❌ No se pudo obtener respuesta válida del modelo {current_model_sdk} tras todos los intentos.")
                # Continuar al siguiente modelo en model_names_to_try_ordered

    # ----- Fin del bucle de modelos -----

    print(f"{log_prefix} ===== RESUMEN DE INTENTO DE MODELOS =====")
    if ai_response_json_payload is not None:
        print(f"{log_prefix} ✅ Respuesta exitosa obtenida")
    else:
        print(f"{log_prefix} ❌ No se pudo obtener respuesta válida de ningún modelo")
        print(f"{log_prefix} Último error: {last_overall_error_type}")
    print(f"{log_prefix} =========================================")

    actions_payload_for_whatsapp = {}
    estado_conv_from_ai = "error_ia_fallback"

    if ai_response_json_payload is None:
        fallback_msg = f"Lo siento, error al procesar con IA ({last_overall_error_type}). Intenta más tarde. ({idioma})"
        actions_payload_for_whatsapp = {
            "1": {"tipo": "mensaje", "mensaje": fallback_msg}}
        ai_response_json_payload_for_history = {
            "estado_conversacion": estado_conv_from_ai,
            **actions_payload_for_whatsapp
        }
        print(f"{log_prefix} Usando respuesta de fallback general. Último error: {last_overall_error_type}")
    else:
        estado_conv_from_ai = ai_response_json_payload.get(
            "estado_conversacion", "desconocido_post_procesado")
        actions_payload_for_whatsapp = {
            k: v for k, v in ai_response_json_payload.items() if k.isdigit()
        }
        ai_response_json_payload_for_history = ai_response_json_payload

    client_msg_timestamp = int(time.time() * 1000)
    upgraded_previous_entry = False
    if history_list_from_file:
        last_entry = history_list_from_file[-1]
        if last_entry.get("role") == "usuario_pausado":
            separator_hist_check = ", "
            fragments_from_last_entry = [text.strip() for text in last_entry.get(
                "mensaje", "").strip().split(separator_hist_check) if text.strip()]
            current_fragments_for_upgrade_check = [f.strip(
            ) for f in base_text_from_fragments.split(separator_hist_check) if f.strip()]

            if fragments_from_last_entry == current_fragments_for_upgrade_check and fragments_from_last_entry:
                time_diff_ms = abs(last_entry.get(
                    "timestamp_ms", 0) - client_msg_timestamp)
                max_expected_diff_ms = (req.delay_seconds + 45) * 1000
                if time_diff_ms < max_expected_diff_ms:
                    print(
                        f"{log_prefix} Se encontró entrada 'usuario_pausado' coincidente. Actualizando a 'cliente'.")
                    last_entry["role"] = "cliente"
                    last_entry["mensaje"] = client_message_for_history
                    last_entry["timestamp_ms"] = client_msg_timestamp
                    if detected_media_type and media_url_from_endpoint:
                        last_entry["media_url"] = media_url_from_endpoint
                        last_entry["media_type"] = detected_media_type
                    else:
                        last_entry.pop("media_url", None)
                        last_entry.pop("media_type", None)
                    upgraded_previous_entry = True

    if not upgraded_previous_entry:
        client_history_entry = {
            "role": "cliente", "mensaje": client_message_for_history, "timestamp_ms": client_msg_timestamp
        }
        if detected_media_type and media_url_from_endpoint:
            client_history_entry["media_url"] = media_url_from_endpoint
            client_history_entry["media_type"] = detected_media_type
        history_list_from_file.append(client_history_entry)

    ai_messages_for_history_log = []
    if isinstance(actions_payload_for_whatsapp, dict):
        try:
            for key_val in sorted(actions_payload_for_whatsapp.keys(), key=lambda k: (int(k) if isinstance(k, str) and k.isdigit() else float('inf'), k)):
                action = actions_payload_for_whatsapp.get(key_val)
                if isinstance(action, dict) and action.get("tipo") == "mensaje":
                    msg_text = action.get("mensaje")
                    if msg_text and isinstance(msg_text, str):
                        ai_messages_for_history_log.append(msg_text.strip())
        except Exception as e_sum_detail:
            print(f"{log_prefix} Error building AI message detail: {e_sum_detail}")
    ai_summary_for_history = "\n".join(
        ai_messages_for_history_log) if ai_messages_for_history_log else "(Respuesta IA no textual, vacía o con error de formato)"

    history_list_from_file.append({
        "role": "asistente",
        "mensaje": ai_summary_for_history,
        "raw_ai_response_payload": ai_response_json_payload_for_history,
        "estado_conversacion": estado_conv_from_ai,
        "timestamp_ms": int(time.time() * 1000)
    })
    try:
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history_list_from_file, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"{log_prefix} Error updating JSON history (AI): {e}")

    return {
        "actions": actions_payload_for_whatsapp,
        "estado_conversacion": estado_conv_from_ai
    }


async def send_whatsapp(server: str, userbot: str, token: str, phone: str, payload: Dict[str, Any]):
    present_url = f"{server}/chats/send-presence?id={userbot}"
    send_url = f"{server}/chats/send?id={userbot}"
    headers = {"Content-Type": "application/json", "x-access-token": token}
    log_prefix = f"[{userbot}/{phone}]"
    if not isinstance(payload, dict) or not all(k.isdigit() for k in payload.keys()):
        print(f"{log_prefix} Error: Payload to send_whatsapp debe ser dict con claves numéricas. Payload: {payload}")
        return
    async with httpx.AsyncClient() as client:
        try:
            await client.post(present_url, headers=headers, json={"receiver": phone, "presence": "composing", "isGroup": False}, timeout=10.0)
        except Exception as e:
            print(f"{log_prefix} Warn: Could not send initial 'composing': {e}")
        payload_keys_sorted = sorted(payload.keys(), key=lambda k: (
            int(k) if isinstance(k, str) and k.isdigit() else float('inf'), k))
        body_base = {"receiver": phone, "isGroup": False}
        for i, key in enumerate(payload_keys_sorted):
            item = payload.get(key)
            if not isinstance(item, dict):
                print(
                    f"{log_prefix} Warn: Item for key '{key}' not dict. Skipping. Item: {item}")
                continue
            tipo = item.get("tipo")
            msg_to_send_content = {}
            current_presence_before_send = None
            try:
                if tipo == "mensaje":
                    msg_to_send_content = {"message": {
                        "text": item.get("mensaje", "(Mensaje vacío)")}}
                elif tipo == "imagen":
                    if not item.get("ruta2"):
                        print(f"{log_prefix} Skipping image send, no ruta2.")
                        continue
                    msg_to_send_content = {"message": {
                        "image": {"url": item.get("ruta2")}, "caption": item.get("mensaje", "")}}
                elif tipo == "video":
                    if not item.get("ruta2"):
                        print(f"{log_prefix} Skipping video send, no ruta2.")
                        continue
                    msg_to_send_content = {"message": {
                        "video": {"url": item.get("ruta2")}, "caption": item.get("mensaje", "")}}
                    current_presence_before_send = "recording"
                elif tipo == "audio":
                    if not item.get("ruta2"):
                        print(f"{log_prefix} Skipping audio send, no ruta2.")
                        continue
                    msg_to_send_content = {"message": {
                        "audio": {"url": item.get("ruta2")}, "ptt": item.get("ptt", True)}}
                    current_presence_before_send = "recording"
                elif tipo == "pdf":
                    if not item.get("ruta2"):
                        print(f"{log_prefix} Skipping pdf send, no ruta2.")
                        continue
                    msg_to_send_content = {"message": {"document": {"url": item.get(
                        "ruta2"), "mimetype": "application/pdf", "fileName": item.get("nombrearchivo", "documento.pdf")}, "caption": item.get("mensaje", "")}}
                elif tipo == "ubicacion":
                    try:
                        lat = float(item.get("lat", 0))
                        long = float(item.get("long", 0))
                        msg_to_send_content = {"message": {"location": {
                            "degreesLatitude": lat, "degreesLongitude": long}}}
                    except (ValueError, TypeError):
                        print(
                            f"{log_prefix} Skipping location, invalid lat/long: {item}")
                        continue
                else:
                    print(
                        f"{log_prefix} Warn: Unknown message type '{tipo}'. Skipping. Item: {item}")
                    continue
                if current_presence_before_send:
                    try:
                        await client.post(present_url, headers=headers, json={**body_base, "presence": current_presence_before_send}, timeout=10.0)
                    except Exception as e_pres:
                        print(
                            f"{log_prefix} Warn: Could not send presence '{current_presence_before_send}': {e_pres}")
                full_send_payload = {**body_base, **msg_to_send_content}
                response = await client.post(send_url, headers=headers, json=full_send_payload, timeout=45.0)
                response.raise_for_status()
                if current_presence_before_send == "recording":
                    next_presence = "composing" if i < len(
                        payload_keys_sorted) - 1 else "paused"
                    try:
                        await client.post(present_url, headers=headers, json={**body_base, "presence": next_presence}, timeout=10.0)
                    except Exception as e_pres_after:
                        print(
                            f"{log_prefix} Warn: Could not send presence '{next_presence}' post-recording: {e_pres_after}")
                # MODIFICACIÓN: Cambio del delay entre mensajes a 0 segundos
                if i < len(payload_keys_sorted) - 1:
                    delay_after = item.get("delay_after_seconds")
                    # Si se especifica un delay, usarlo. Si no, usar 0 segundos en lugar de 1.5
                    await asyncio.sleep(delay_after if isinstance(delay_after, (int, float)) and delay_after > 0 else 0)
            except Exception as e_send:
                print(
                    f"{log_prefix} Error sending message type '{tipo}' (key {key}): {e_send}")
        try:
            await client.post(present_url, headers=headers, json={**body_base, "presence": "paused"}, timeout=10.0)
        except Exception as e_final_pres:
            print(
                f"{log_prefix} Warn: Could not send final 'paused' presence: {e_final_pres}")


async def send_notification(server: str, userbot: str, token: str, receiver: str, is_group: bool, linea_wa: str, notification_content=None, log_prefix: str = ""):
    """
    Envía una notificación al endpoint POST /chats/send cuando se completa un estado específico.
    Envía un solo mensaje combinado que contiene:
    1. El detalle del pedido.
    2. El enlace del pedido (usuario) al final.
    """
    send_url = f"{server}/chats/send?id={userbot}"
    headers = {"Content-Type": "application/json", "x-access-token": token}

    # Construir el mensaje de notificación (solo el enlace)
    link_message_text = f"https://wa.me/{linea_wa}"

    # Construir el mensaje con el detalle de la IA
    detail_message_text = "(No se pudo extraer el detalle del pedido de la IA)"

    if notification_content:
        if isinstance(notification_content, str):
            detail_message_text = notification_content
        elif isinstance(notification_content, dict):
            mensajes_ia = []
            actions_dict = notification_content.get(
                "actions", notification_content)

            for key, value in actions_dict.items():
                if str(key).isdigit() and isinstance(value, dict) and value.get("tipo") == "mensaje":
                    if value.get("mensaje"):
                        mensajes_ia.append(value.get("mensaje"))

            if mensajes_ia:
                detail_message_text = "Detalle del pedido:\n\n" + \
                    "\n\n".join(mensajes_ia)

    # Combinar ambos en un solo texto
    combined_message_text = f"{detail_message_text}\n\n{link_message_text}"

    combined_payload = {
        "receiver": receiver,
        "isGroup": is_group,
        "message": {
            "text": combined_message_text
        }
    }

    try:
        async with httpx.AsyncClient() as client:
            print(
                f"{log_prefix} Enviando notificación combinada (detalle + enlace) a {receiver} (grupo: {is_group})")

            # Enviar el mensaje único
            resp = await client.post(send_url, headers=headers, json=combined_payload, timeout=30.0)
            resp.raise_for_status()
            print(
                f"{log_prefix} Notificación combinada enviada exitosamente a {receiver}")

            return True
    except Exception as e:
        print(f"{log_prefix} Error enviando notificación combinada a {receiver}: {e}")
        return False


async def extract_order_info_with_ai(userbot: str, phone: str, apikey: str, model: str) -> Optional[Dict[str, Any]]:
    history_file = hist_file_path(userbot, phone)
    history_list = []
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history_list = json.load(f)
        except Exception:
            pass

    if not history_list:
        return None

    history_text = ""
    for item in history_list[-20:]:
        role = item.get("role", "desconocido")
        msg = item.get("mensaje", "")
        history_text += f"{role}: {msg}\n"

    prompt = """
    Extrae la siguiente información de la conversación para un pedido a domicilio.
    Devuelve un JSON estricto con esta estructura, sin bloques de código ni markdown ni texto adicional:
    {
      "resumen": "Resumen breve de la conversación",
      "pedido": "Breve resumen de lo pedido",
      "detalle_completo": "Lista EXACTA y completa de todos los productos pedidos con sus respectivos precios, tal cual como el chatbot se lo detalló al cliente",
      "total_a_cobrar": "Valor numérico o en texto del total de los productos SIN sumar el valor del domicilio (ej. $35.000)",
      "direccion": "Dirección de entrega especificada (o 'No especificada')",
      "metodo_pago": "Método de pago (ej. 'Efectivo', 'Transferencia', o 'No especificado')"
    }
    Conversación:
    """ + history_text

    model_names_to_try = []
    if model and model.strip():
        model_names_to_try.append(model.strip())

    for fallback in GEMINI_FALLBACK_MODELS_LIST:
        if fallback not in model_names_to_try:
            model_names_to_try.append(fallback)

    if "gemini-2.0-flash" not in model_names_to_try:
        model_names_to_try.append("gemini-2.0-flash")

    raw_text = None
    last_error = None
    loop = asyncio.get_event_loop()

    for current_model in model_names_to_try:
        try:
            raw_text = await loop.run_in_executor(
                None,
                _call_gemini_sdk_sync,
                apikey,
                prompt,
                "Eres un asistente que extrae información de pedidos en JSON estrictamente.",
                current_model,
                0.1, 0.95, 65536,
                f"[{userbot}/{phone}/extract]",
                "UTC", "UTC", "", 0, "HIGH", "MEDIA_RESOLUTION_HIGH", "", False, False
            )

            cleaned = raw_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            data = json.loads(cleaned, strict=False)
            return data  # Exito total
        except Exception as e:
            last_error = e
            print(
                f"[{userbot}/{phone}/extract] Error con modelo {current_model} (Extraccion o Parseo JSON): {e}")
            continue

    print(f"[{userbot}/{phone}/extract] Todos los modelos fallaron para la extraccion o el parseo JSON. Ultimo error: {last_error}")
    return None


async def delayed_processing_task(task_key: str):
    if task_key not in processing_tasks:
        return
    task_info = processing_tasks[task_key]
    current_task_req_info: MessageRequest = task_info['original_request']
    log_prefix = f"[{current_task_req_info.userbot}/{current_task_req_info.lineaWA}]"

    try:
        print(f"{log_prefix} Iniciando retraso de {current_task_req_info.delay_seconds}s para tarea {task_key}...")
        push_name_log = task_info.get('user_push_name', current_task_req_info.userbot)
        print(f"{log_prefix} Procesando en: {current_task_req_info.delay_seconds} el de {push_name_log}")
        await asyncio.sleep(current_task_req_info.delay_seconds)

        if task_key not in processing_tasks or processing_tasks[task_key]['task'] is not asyncio.current_task():
            print(
                f"{log_prefix} Tarea {task_key} (esta instancia) fue cancelada o reemplazada. Abortando.")
            return

        current_state_after_delay = load_contact_state(
            current_task_req_info.userbot, current_task_req_info.lineaWA)
        if is_contact_paused(current_state_after_delay, current_task_req_info.pause_timeout_minutes):
            print(
                f"{log_prefix} Contacto PAUSADO detectado después del retraso para tarea {task_key}.")
            await append_paused_history(current_task_req_info)
            return

        print(
            f"{log_prefix} Contacto NO pausado para tarea {task_key}. Procesando mensaje normalmente.")

        if task_key not in processing_tasks or processing_tasks[task_key]['task'] is not asyncio.current_task():
            print(
                f"{log_prefix} Tarea {task_key} (esta instancia) desactualizada post-sincronización. Abortando.")
            return

        task_info = processing_tasks[task_key]
        actual_fragments = [f.strip()
                            for f in task_info['fragments'] if f.strip()]
        # Obtener el pushName almacenado
        user_push_name = task_info.get('user_push_name', '')

        if not actual_fragments:
            print(
                f"{log_prefix} Tarea {task_key}: No hay fragmentos válidos para enviar a la IA.")
            return

        print(f"{log_prefix} Tarea {task_key}: Retraso completado. Fragmentos: {actual_fragments}. Procesando...")
        # No se pasa ai_model aquí, ya que process_message_final usa req.ai_model y la lista de fallback
        ai_response = await process_message_final(
            current_task_req_info, actual_fragments,
            current_task_req_info.pais, current_task_req_info.idioma,
            user_push_name  # Pasar el pushName a process_message_final
        )

        if ai_response and isinstance(ai_response, dict):
            ai_actions_payload = ai_response.get("actions", {})
            estado_conversacion_ai = ai_response.get("estado_conversacion", "")

            # Lógica de notificación
            if (current_task_req_info.activarnotificacion and
                current_task_req_info.estado and
                current_task_req_info.lineaogruponotificacion and
                    estado_conversacion_ai == current_task_req_info.estado):

                # --- Lógica de control de duplicados ---
                notif_cache_key = f"{current_task_req_info.userbot}_{current_task_req_info.lineaWA}"
                current_ts = time.time()
                should_notify = True

                # Configurar tiempo de re-notificación (ej. 10 minutos = 600 segundos)
                RENOTIFICATION_TIMEOUT_SECONDS = 600

                if notif_cache_key in notification_state_cache:
                    last_notif_data = notification_state_cache[notif_cache_key]
                    last_state = last_notif_data.get("last_state")
                    last_ts = last_notif_data.get("timestamp", 0)

                    if last_state == estado_conversacion_ai:
                        time_diff = current_ts - last_ts
                        if time_diff < RENOTIFICATION_TIMEOUT_SECONDS:
                            print(
                                f"{log_prefix} Estado '{estado_conversacion_ai}' ya notificado hace {int(time_diff)}s. Omitiendo duplicado.")
                            should_notify = False
                        else:
                            print(
                                f"{log_prefix} Estado '{estado_conversacion_ai}' re-notificando tras {int(time_diff)}s (timeout superado).")

                if should_notify:
                    # Determinar qué userbot usar para la notificación
                    userbot_notificacion = current_task_req_info.userbot
                    if current_task_req_info.activaruserbotopcional and current_task_req_info.userbotopcional:
                        userbot_notificacion = current_task_req_info.userbotopcional
                        print(
                            f"{log_prefix} Usando userbot opcional para notificación: {userbot_notificacion}")

                    print(
                        f"{log_prefix} Estado de conversación coincide ('{estado_conversacion_ai}'). Extrayendo datos con IA antes de notificar...")
                    try:
                        order_info = await extract_order_info_with_ai(current_task_req_info.userbot, current_task_req_info.lineaWA, current_task_req_info.apikey, current_task_req_info.ai_model)

                        notification_text = ai_response  # Fallback

                        if order_info:
                            try:
                                orders_data = {}
                                if os.path.exists(ORDERS_FILE_PATH):
                                    with open(ORDERS_FILE_PATH, 'r', encoding='utf-8') as f:
                                        try:
                                            orders_data = json.load(f)
                                        except Exception:
                                            orders_data = {}

                                order_timestamp = int(time.time() * 1000)
                                order_id = current_task_req_info.lineaWA

                                # --- NUEVO: Control estricto de duplicados por contenido del pedido ---
                                is_exact_duplicate = False
                                if order_id in orders_data:
                                    existing_order = orders_data[order_id]
                                    # Solo comparar si el pedido existente tiene menos de 12 horas
                                    if order_timestamp - existing_order.get("timestamp", 0) < 12 * 3600 * 1000:
                                        def clean_str(t):
                                            if not t:
                                                return ""
                                            return "".join(c.lower() for c in str(t) if c.isalnum())

                                        existing_det = clean_str(
                                            existing_order.get("detalle_completo", ""))
                                        existing_tot = clean_str(
                                            existing_order.get("total_a_cobrar", ""))
                                        new_det = clean_str(
                                            order_info.get("detalle_completo", ""))
                                        new_tot = clean_str(
                                            order_info.get("total_a_cobrar", ""))

                                        if existing_det == new_det and existing_tot == new_tot:
                                            is_exact_duplicate = True

                                if is_exact_duplicate:
                                    print(
                                        f"{log_prefix} El detalle y total del pedido no han cambiado. Omitiendo notificación duplicada.")
                                    # No guardamos nada en orders.json ni enviamos mensaje de WhatsApp de notificación
                                else:
                                    # Actualizar caché de notificación
                                    notification_state_cache[notif_cache_key] = {
                                        "last_state": estado_conversacion_ai,
                                        "timestamp": current_ts
                                    }

                                    is_modification = False
                                    if order_id in orders_data and (order_timestamp - orders_data[order_id].get("timestamp", 0) < 12 * 3600 * 1000) and orders_data[order_id].get("status") != "Despachados":
                                        # Es un pedido duplicado/reciente del mismo cliente ANTES de ser despachado. Actualizamos info pero mantenemos el historial/reloj.
                                        orders_data[order_id]["summary"] = order_info.get(
                                            "resumen", "")
                                        orders_data[order_id]["pedido"] = order_info.get(
                                            "pedido", "")
                                        orders_data[order_id]["detalle_completo"] = order_info.get(
                                            "detalle_completo", "")
                                        orders_data[order_id]["total_a_cobrar"] = order_info.get(
                                            "total_a_cobrar", "")
                                        orders_data[order_id]["direccion"] = order_info.get(
                                            "direccion", "")
                                        orders_data[order_id]["metodo_pago"] = order_info.get(
                                            "metodo_pago", "")
                                        is_modification = True
                                        print(
                                            f"{log_prefix} Actualizando pedido existente para evitar duplicados en el dashboard.")
                                    else:
                                        # Es un pedido completamente nuevo
                                        orders_data[order_id] = {
                                            "phone": current_task_req_info.lineaWA,
                                            "summary": order_info.get("resumen", ""),
                                            "pedido": order_info.get("pedido", ""),
                                            "detalle_completo": order_info.get("detalle_completo", ""),
                                            "total_a_cobrar": order_info.get("total_a_cobrar", ""),
                                            "direccion": order_info.get("direccion", ""),
                                            "metodo_pago": order_info.get("metodo_pago", ""),
                                            "status": "Recientes",
                                            "timestamp": order_timestamp,
                                            "static_duration": 0,
                                            "history": [{"status": "Recientes", "timestamp": order_timestamp}]
                                        }

                                    with open(ORDERS_FILE_PATH, 'w', encoding='utf-8') as f:
                                        json.dump(orders_data, f,
                                                  ensure_ascii=False, indent=4)
                                    print(
                                        f"{log_prefix} Pedido guardado en orders.json exitosamente.")

                                    # Cabecera dinámica según si es nuevo o modificado
                                    header_title = "🚨 *NUEVO PEDIDO* 🚨"
                                    if is_modification:
                                        header_title = "⚠️ *PEDIDO MODIFICADO* ⚠️"

                                    notification_text = (
                                        f"{header_title}\n\n"
                                        f"📱 *Cliente:* {current_task_req_info.lineaWA}\n\n"
                                        f"📝 *Detalle del Pedido:*\n{order_info.get('detalle_completo', 'No especificado')}\n\n"
                                        f"💰 *Total a Cobrar:* {order_info.get('total_a_cobrar', 'No especificado')}\n"
                                        f"💳 *Método de Pago:* {order_info.get('metodo_pago', 'No especificado')}\n"
                                        f"📍 *Dirección de Entrega:* {order_info.get('direccion', 'No especificada')}\n\n"
                                        f"🤖 *Resumen IA:* {order_info.get('resumen', '')}"
                                    )

                                    # Enviar la notificación
                                    success = await send_notification(
                                        current_task_req_info.server,
                                        userbot_notificacion,
                                        current_task_req_info.token,
                                        current_task_req_info.lineaogruponotificacion,
                                        current_task_req_info.lineaogrupo or False,
                                        current_task_req_info.lineaWA,
                                        notification_text,
                                        log_prefix
                                    )
                                    if success:
                                        print(
                                            f"{log_prefix} Notificación enviada exitosamente con detalle completo.")
                            except Exception as e:
                                print(
                                    f"{log_prefix} Error armando notificación y guardando pedido: {e}")
                    except Exception as e:
                        print(
                            f"{log_prefix} Error en el proceso de notificación y extracción: {e}")
            else:
                # Si el estado cambió a algo diferente del objetivo, podemos limpiar el caché o dejarlo
                # para que si vuelve al estado objetivo, se considere "nuevo" cambio de estado.
                # Estrategia: Si el estado actual NO es el de notificación, limpiamos el caché para permitir
                # notificar inmediatamente si vuelve a entrar en el estado objetivo.
                notif_cache_key = f"{current_task_req_info.userbot}_{current_task_req_info.lineaWA}"
                if notif_cache_key in notification_state_cache and notification_state_cache[notif_cache_key]["last_state"] == current_task_req_info.estado:
                    # Si teníamos guardado el estado objetivo y ahora la IA dice otro estado, borramos caché
                    # para que la próxima vez que diga el estado objetivo, notifique de nuevo.
                    if estado_conversacion_ai != current_task_req_info.estado:
                        del notification_state_cache[notif_cache_key]
                        print(
                            f"{log_prefix} Estado cambió de '{current_task_req_info.estado}' a '{estado_conversacion_ai}'. Reset de control de duplicados.")

            if ai_actions_payload:
                await send_whatsapp(
                    current_task_req_info.server, current_task_req_info.userbot,
                    current_task_req_info.token, current_task_req_info.lineaWA,
                    ai_actions_payload
                )
            else:
                print(
                    f"{log_prefix} Tarea {task_key}: No se obtuvo payload de acciones de la IA.")
        else:
            print(
                f"{log_prefix} Tarea {task_key}: No se obtuvo respuesta de la IA (o respuesta inválida).")

        print(f"{log_prefix} Tarea {task_key} completada.")

    except GeminiBlockedError:  # Capturar el re-lanzado desde process_message_final
        print(f"{log_prefix} Tarea {task_key} detenida debido a bloqueo de contenido de Gemini. No se enviará respuesta.")
    except asyncio.CancelledError:
        print(
            f"{log_prefix} Tarea de procesamiento retrasado ({task_key}) fue cancelada explícitamente.")
    except Exception as e:
        print(
            f"{log_prefix} Error en tarea de procesamiento retrasado ({task_key}): {e}")
        import traceback
        traceback.print_exc()
    finally:
        if task_key in processing_tasks and processing_tasks.get(task_key, {}).get('task') is asyncio.current_task():
            del processing_tasks[task_key]
            print(
                f"{log_prefix} Tarea {task_key} (esta instancia) finalizada y eliminada del registro.")


@app.post("/wa/process")
async def handle_incoming_message(req: MessageRequest):
    phone, userbot = req.lineaWA, req.userbot
    if getattr(req, "activaruserbotopcional", False) and getattr(req, "userbotopcional", None):
        userbot = req.userbotopcional.strip()

    new_fragment = req.mensaje_reciente.strip(
    ) if req.mensaje_reciente and isinstance(req.mensaje_reciente, str) else ""
    task_key = f"{userbot}_{phone}"
    log_prefix = f"[{userbot}/{phone}]"

    # 🔧 DEBUG: Imprimir configuración de userbot opcional recibida
    print(f"{log_prefix} 📨 Request recibido en /wa/process")
    print(f"{log_prefix}    - activaruserbotopcional: {req.activaruserbotopcional}")
    print(f"{log_prefix}    - userbotopcional: {req.userbotopcional}")
    print(f"{log_prefix}    - activarnotificacion: {req.activarnotificacion}")
    print(f"{log_prefix}    - estado para notificar: {req.estado}")
    print(f"{log_prefix}    - lineaogruponotificacion: {req.lineaogruponotificacion}")

    required_fields_map = {
        "lineaWA": req.lineaWA, "userbot": req.userbot,
        "apikey": req.apikey, "server": req.server, "promt": req.promt, "token": req.token,
        "pais": req.pais, "idioma": req.idioma, "ai_model": req.ai_model
    }
    missing_or_empty_fields = []
    for k, v in required_fields_map.items():
        if not v or (isinstance(v, str) and not v.strip()):
            missing_or_empty_fields.append(k)

    if missing_or_empty_fields:
        raise HTTPException(
            status_code=400, detail=f"Campos requeridos faltantes o vacíos: {', '.join(missing_or_empty_fields)}")

    # === AUTO-GUARDADO DE BOT EN BASE DE DATOS ===
    try:
        from core.database import SessionLocal
        from models import bot_models
        db = SessionLocal()
        db_bot = db.query(bot_models.BotConfig).filter(
            bot_models.BotConfig.userbot_identifier == userbot).first()
        if not db_bot:
            new_bot = bot_models.BotConfig(
                user_id=None,  # Queda huérfano hasta que un cliente lo reclame en el panel
                userbot_identifier=userbot,
                apikey=req.apikey,
                system_prompt=req.promt,
                ai_model=req.ai_model,
                thinking_budget=getattr(req, 'thinking_budget', -1),
                thinking_level=getattr(req, 'thinking_level', 'HIGH'),
                pais=req.pais,
                idioma=req.idioma,
                delay_seconds=req.delay_seconds,
                pause_timeout_minutes=req.pause_timeout_minutes,
                activarnotificacion=req.activarnotificacion,
                estado_notificacion=req.estado,
                lineaogruponotificacion=req.lineaogruponotificacion,
                activaruserbotopcional=req.activaruserbotopcional,
                userbotopcional=req.userbotopcional
            )
            db.add(new_bot)
            db.commit()
            print(
                f"{log_prefix} 🤖 [Auto-Save] Nueva configuración de bot guardada (huérfana) en SQLite para el ID: {userbot}")
        db.close()
    except Exception as e:
        print(f"{log_prefix} ❌ Error en auto-guardado de BD: {e}")

    # Validaciones estrictas de compatibilidad de parámetros por modelo
    is_gemini_3 = "gemini-3" in req.ai_model

    if is_gemini_3:
        # Para modelos 3.0+, thinking_budget DEBE ser 0 (desactivado)
        if hasattr(req, 'thinking_budget') and req.thinking_budget not in [0, -1]:
            # Solo loggeamos la advertencia pero lo forzamos a 0 más adelante en la inyección
            print(f"{log_prefix} ADVERTENCIA: Se recibió thinking_budget={req.thinking_budget} para modelo {req.ai_model}. Este parámetro no es compatible con modelos 3.0+ y será ignorado a favor de thinking_level.")

        # Log de información sobre configuración de razonamiento para 3.0+
        level_to_use = getattr(req, 'thinking_level', 'HIGH')
        res_to_use = getattr(req, 'media_resolution', 'MEDIA_RESOLUTION_HIGH')
        print(f"{log_prefix} Modelo {req.ai_model} (Familia 3.0+). Usando: ThinkingLevel={level_to_use}, MediaRes={res_to_use}")
    else:
        # Para modelos < 3.0
        if hasattr(req, 'thinking_budget') and req.thinking_budget not in [-1, 0] and req.thinking_budget < 0:
            raise HTTPException(
                status_code=400, detail="thinking_budget debe ser -1 (ilimitado), 0 (deshabilitado) o un entero positivo (límite específico de tokens)")

        # Log de información sobre thinking_config para < 3.0
        if req.ai_model in THINKING_COMPATIBLE_MODELS:
            budget_val = getattr(req, 'thinking_budget', -1)
            thinking_status = "habilitado" if budget_val != 0 else "deshabilitado"
            print(f"{log_prefix} Modelo {req.ai_model} compatible con thinking_config. Razonamiento {thinking_status} (budget: {budget_val})")
        else:
            if hasattr(req, 'thinking_budget') and req.thinking_budget != 0:
                print(f"{log_prefix} ADVERTENCIA: thinking_budget={req.thinking_budget} especificado pero modelo {req.ai_model} no es compatible. Se ignorará el parámetro.")

    current_state = load_contact_state(userbot, phone)
    was_previously_paused = is_contact_paused(
        current_state, req.pause_timeout_minutes)

    # Obtener timestamps de reacciones de control Y el pushName del usuario
    latest_pause_ts_ms, latest_unpause_ts_ms, user_push_name = await get_latest_control_reaction_timestamps_and_push_name(userbot, req.token, phone, req.server)

    new_state_determined = current_state.copy()
    last_ctl_react_ts_in_state = current_state.get(
        "last_control_reaction_timestamp", 0)
    abs_latest_ctl_ts_from_api = max(latest_pause_ts_ms, latest_unpause_ts_ms)
    state_changed_by_reaction_or_timeout = False

    if abs_latest_ctl_ts_from_api > 0 and abs_latest_ctl_ts_from_api > last_ctl_react_ts_in_state:
        print(f"{log_prefix} Nueva reacción de control detectada.")
        if latest_unpause_ts_ms > latest_pause_ts_ms:
            if new_state_determined["is_paused"]:
                state_changed_by_reaction_or_timeout = True
            new_state_determined["is_paused"] = False
            new_state_determined["pause_start_time"] = None
            new_state_determined["last_control_reaction_timestamp"] = latest_unpause_ts_ms
            print(f"{log_prefix} Despausando por reacción ✅.")
        else:
            # SIEMPRE reiniciar el timer de pausa cuando hay una nueva reacción ✋,
            # incluso si ya estaba "pausado" teóricamente
            state_changed_by_reaction_or_timeout = True
            now_iso = datetime.now(dt_timezone.utc).isoformat()
            new_state_determined["pause_start_time"] = now_iso
            new_state_determined["last_message_timestamp_during_pause_ms"] = int(
                datetime.fromisoformat(now_iso).timestamp() * 1000)

            new_state_determined["is_paused"] = True
            new_state_determined["last_control_reaction_timestamp"] = latest_pause_ts_ms
            print(f"{log_prefix} Pausando por reacción ✋.")

    if new_state_determined["is_paused"] and new_state_determined["pause_start_time"] and req.pause_timeout_minutes > 0:
        try:
            pause_start_dt = datetime.fromisoformat(
                new_state_determined["pause_start_time"])
            if pause_start_dt.tzinfo is None:
                pause_start_dt = pause_start_dt.replace(tzinfo=dt_timezone.utc)
            if datetime.now(dt_timezone.utc) - pause_start_dt > timedelta(minutes=req.pause_timeout_minutes):
                if new_state_determined["is_paused"]:
                    new_state_determined["is_paused"] = False
                    new_state_determined["pause_start_time"] = None
                    state_changed_by_reaction_or_timeout = True
                    print(f"{log_prefix} Despausando por timeout.")
        except Exception as e:
            print(f"{log_prefix} Error procesando timeout de pausa: {e}")

    if state_changed_by_reaction_or_timeout or new_state_determined != current_state:
        save_contact_state(userbot, phone, new_state_determined)
        current_state = new_state_determined

    is_truly_paused_now = is_contact_paused(
        current_state, req.pause_timeout_minutes)

    if was_previously_paused and not is_truly_paused_now:
        print(
            f"{log_prefix} Contacto despausado. Sincronizando historial de pausa final.")
        await append_paused_history(req)

    if is_truly_paused_now:
        print(
            f"{log_prefix} Contacto está PAUSADO. Mensaje '{new_fragment[:30]}...' se tratará como parte de la pausa.")
        if task_key in processing_tasks:
            task_to_cancel = processing_tasks[task_key].get('task')
            if task_to_cancel and not task_to_cancel.done():
                task_to_cancel.cancel()
                print(
                    f"{log_prefix} 🛑 Generación de IA o retraso interrumpido y cancelado por reacción ✋.")
            # Limpiar la tarea para asegurar el estado del contenedor
            del processing_tasks[task_key]
        paused_hist_res = await append_paused_history(req)
        return {"status": "contact_paused", "detail": f"Contacto {log_prefix} pausado . {paused_hist_res.get('detail', '')}"}

    if not new_fragment:
        print(f"{log_prefix} Mensaje reciente vacío y contacto no pausado. No se (re)programa tarea de IA con este fragmento.")
        if task_key in processing_tasks:
            return {"status": "received_empty_fragment_task_exists", "detail": f"{log_prefix} Mensaje vacío, tarea existente no modificada por este fragmento."}
        else:
            return {"status": "received_empty_fragment_not_paused", "detail": f"{log_prefix} Mensaje vacío, no programado para IA."}

    if task_key in processing_tasks:
        task_info = processing_tasks[task_key]
        old_task_future = task_info.get('task')

        if old_task_future and not old_task_future.done():
            print(
                f"{log_prefix} Tarea IA existente para {task_key} encontrada. Cancelando y reiniciando timer.")
            old_task_future.cancel()

        task_info['fragments'].append(new_fragment)
        task_info['original_request'] = req
        # Almacenar el pushName en task_info
        task_info['user_push_name'] = user_push_name

        new_delayed_task = asyncio.create_task(
            delayed_processing_task(task_key))
        processing_tasks[task_key] = {
            'task': new_delayed_task,
            'fragments': task_info['fragments'],
            'original_request': req,
            'user_push_name': user_push_name  # Almacenar el pushName en task_info
        }
        print(
            f"{log_prefix} Tarea IA actualizada/reemplazada para {task_key} con fragmento '{new_fragment[:30]}...'. Fragmentos totales: {len(task_info['fragments'])}")
    else:
        new_delayed_task = asyncio.create_task(
            delayed_processing_task(task_key))
        processing_tasks[task_key] = {
            'task': new_delayed_task,
            'fragments': [new_fragment],
            'original_request': req,
            'user_push_name': user_push_name  # Almacenar el pushName en task_info
        }
        print(
            f"{log_prefix} Nueva tarea IA para {task_key} con fragmento '{new_fragment[:30]}...'.")

    return {"status": "received", "detail": f"Mensaje para {log_prefix} recibido y programado para procesamiento con delay {req.delay_seconds}s."}


@app.post("/wa/delete-history")
async def handle_delete_history(req: DeleteHistoryRequest):
    userbot = req.userbot
    if not userbot:
        raise HTTPException(
            status_code=400, detail="Userbot es un campo requerido.")
    userbot_hist_dir = os.path.join(HIST_BASE_DIR, userbot)
    userbot_paused_status_dir = os.path.join(PAUSED_STATUS_DIR, userbot)
    results = {"userbot": userbot, "deleted_count": 0,
               "failed_count": 0, "deleted_items": [], "failed_items": {}}
    log_prefix = f"[{userbot}/delete_history]"
    print(f"{log_prefix} Solicitud de eliminación de historial (delete_all: {req.delete_all}).")
    files_to_consider_for_deletion: List[Tuple[str, str, str]] = []
    if req.delete_all:
        if os.path.isdir(userbot_hist_dir):
            files_to_consider_for_deletion.extend([(os.path.join(userbot_hist_dir, f), f.replace(
                '.json', ''), 'historial_json') for f in os.listdir(userbot_hist_dir) if f.endswith('.json')])
        if os.path.isdir(userbot_paused_status_dir):
            files_to_consider_for_deletion.extend([(os.path.join(userbot_paused_status_dir, f), f.replace(
                '.json', ''), 'paused_status_json') for f in os.listdir(userbot_paused_status_dir) if f.endswith('.json')])
    elif req.lineaWAs_to_delete and isinstance(req.lineaWAs_to_delete, list):
        for phone in req.lineaWAs_to_delete:
            if not phone or not isinstance(phone, str):
                continue
            hist_f_json = os.path.join(userbot_hist_dir, f"{phone}.json")
            stat_f_json = os.path.join(
                userbot_paused_status_dir, f"{phone}.json")
            if os.path.exists(hist_f_json):
                files_to_consider_for_deletion.append(
                    (hist_f_json, phone, 'historial_json'))
            if os.path.exists(stat_f_json):
                files_to_consider_for_deletion.append(
                    (stat_f_json, phone, 'paused_status_json'))
    else:
        raise HTTPException(
            status_code=400, detail="Si delete_all es false, se requiere lineaWAs_to_delete.")
    for file_path, phone_key, item_type in files_to_consider_for_deletion:
        try:
            os.remove(file_path)
            results["deleted_items"].append(
                f"{item_type}/{os.path.basename(file_path)}")
            results["deleted_count"] += 1
            cache_key = f"{userbot}_{phone_key}"
            if item_type == 'paused_status_json' and cache_key in contact_pause_state_cache:
                del contact_pause_state_cache[cache_key]
                print(f"{log_prefix} Cache entry {cache_key} eliminada.")
            print(f"{log_prefix} Archivo eliminado: {file_path}")
        except OSError as e:
            results["failed_items"][f"{item_type}/{os.path.basename(file_path)}"] = str(
                e)
            results["failed_count"] += 1
            print(f"{log_prefix} Error al eliminar {file_path}: {e}")
    if results["deleted_count"] == 0 and results["failed_count"] == 0 and not files_to_consider_for_deletion:
        results["status"] = "no_files_found_to_delete"
    elif results["failed_count"] == 0:
        results["status"] = "success"
    else:
        results["status"] = "partial_success"
    results["message"] = f"Operación de eliminación para '{userbot}' completada."
    if not req.delete_all and req.lineaWAs_to_delete:
        results["attempted_lineaWAs_count"] = len(req.lineaWAs_to_delete)
    print(f"{log_prefix} Eliminación finalizada. Resultados: {results}")
    return results


@app.post("/wa/get-history")
async def get_history_endpoint(req: GetHistoryRequest):
    userbot, phone = req.userbot, req.lineaWA
    log_prefix = f"[{userbot}/{phone}/get_history]"
    print(f"{log_prefix} Solicitud historial: role='{req.role_filter}', count={req.count}, keyword='{req.keyword_search}'")
    history_file = hist_file_path(userbot, phone)
    if not os.path.exists(history_file):
        print(f"{log_prefix} Historial no encontrado.")
        return {}
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            history_list: List[Dict[str, Any]] = json.load(f)
        if not isinstance(history_list, list):
            print(f"{log_prefix} Formato historial inválido.")
            return {}
    except Exception as e:
        print(f"{log_prefix} Error cargando historial: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error leyendo historial: {e}")

    filtered_history = history_list
    if req.role_filter:
        role_filter_lower = req.role_filter.lower()
        valid_specific_roles = ["cliente",
                                "usuario_pausado", "asistente", "operador"]
        if role_filter_lower == "cliente":
            filtered_history = [m for m in filtered_history if m.get(
                "role", "").lower() in ["cliente", "usuario_pausado"]]
        elif role_filter_lower == "asistente":
            filtered_history = [m for m in filtered_history if m.get(
                "role", "").lower() in ["asistente", "operador"]]
        elif role_filter_lower in valid_specific_roles:
            filtered_history = [m for m in filtered_history if m.get(
                "role", "").lower() == role_filter_lower]
        else:
            raise HTTPException(
                status_code=400, detail=f"role_filter inválido: {req.role_filter}. Use: cliente (incluye usuario_pausado), asistente (incluye operador), o uno específico de la lista {valid_specific_roles}.")

    if req.keyword_search:
        try:
            regex = re.compile(req.keyword_search, re.IGNORECASE)
            filtered_history = [
                m for m in filtered_history
                if isinstance(m.get("mensaje"), str) and regex.search(m["mensaje"])
            ]
        except re.error as e:
            print(f"{log_prefix} Regex inválido: {e}")
            raise HTTPException(
                status_code=400, detail=f"Regex keyword_search inválido: {e}")

    def sort_key_for_history(message: Dict[str, Any]) -> Tuple[int, int]:
        timestamp = message.get("timestamp_ms", 0)
        if not isinstance(timestamp, (int, float)):
            timestamp = 0

        role = message.get("role", "").lower()
        role_priority = 2
        if role in ["asistente", "operador"]:
            role_priority = 0
        elif role in ["cliente", "usuario_pausado"]:
            role_priority = 1
        return (-int(timestamp), role_priority)

    messages_ordered_for_response = sorted(
        filtered_history, key=sort_key_for_history)

    final_messages_to_return = messages_ordered_for_response
    if req.count is not None and req.count > 0:
        final_messages_to_return = messages_ordered_for_response[:req.count]
    elif req.count == 0:
        final_messages_to_return = messages_ordered_for_response

    response_payload = {str(i + 1): msg for i,
                        msg in enumerate(final_messages_to_return)}
    print(f"{log_prefix} Historial recuperado. Enviando {len(response_payload)} mensajes.")
    return response_payload


@app.get("/api/orders")
async def get_orders():
    if os.path.exists(ORDERS_FILE_PATH):
        try:
            with open(ORDERS_FILE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


@app.put("/api/orders/{order_id}/status")
async def update_order_status(order_id: str, update: OrderStatusUpdate):
    if os.path.exists(ORDERS_FILE_PATH):
        try:
            with open(ORDERS_FILE_PATH, 'r', encoding='utf-8') as f:
                orders = json.load(f)
            if order_id in orders:
                import time
                orders[order_id]["status"] = update.status
                if "history" not in orders[order_id]:
                    orders[order_id]["history"] = [{"status": "Recientes", "timestamp": orders[order_id].get(
                        "timestamp", int(time.time() * 1000))}]
                orders[order_id]["history"].append(
                    {"status": update.status, "timestamp": int(time.time() * 1000)})
                with open(ORDERS_FILE_PATH, 'w', encoding='utf-8') as f:
                    json.dump(orders, f, ensure_ascii=False, indent=4)
                return {"status": "success", "message": "Estado actualizado"}
            else:
                raise HTTPException(
                    status_code=404, detail="Pedido no encontrado")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    raise HTTPException(
        status_code=404, detail="Base de datos de pedidos no encontrada")


@app.get("/dashboard")
async def dashboard():
    html_content = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dashboard de Pedidos WhatsApp</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-color: #0f172a;
                --card-bg: #1e293b;
                --text-main: #f8fafc;
                --text-muted: #94a3b8;
                --primary: #3b82f6;
                --primary-hover: #2563eb;
                --success: #10b981;
                --warning: #f59e0b;
                --whatsapp: #25D366;
                --border: #334155;
                --highlight: #f43f5e;
            }
            body {
                font-family: 'Inter', sans-serif;
                background-color: var(--bg-color);
                color: var(--text-main);
                margin: 0;
                padding: 0;
                display: flex;
                flex-direction: column;
                height: 100vh; /* Fallback */
                height: 100dvh; /* Dynamic viewport height para móviles */
                overflow: hidden;
            }
            header {
                background: var(--card-bg);
                padding: 1.2rem 2rem;
                border-bottom: 1px solid var(--border);
                display: flex;
                justify-content: space-between;
                align-items: center;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                z-index: 10;
            }
            h1 { margin: 0; font-size: 1.4rem; font-weight: 600; }
            .header-controls {
                display: flex;
                align-items: center;
                gap: 1.5rem;
            }
            .date-picker {
                background: var(--bg-color);
                color: var(--text-main);
                border: 1px solid var(--border);
                padding: 0.5rem 1rem;
                border-radius: 6px;
                font-family: 'Inter', sans-serif;
                outline: none;
                cursor: pointer;
            }
            .date-picker::-webkit-calendar-picker-indicator {
                filter: invert(1);
                cursor: pointer;
            }
            .tabs {
                display: flex;
                gap: 0.5rem;
                padding: 1rem 2rem;
                background: var(--card-bg);
                border-bottom: 1px solid var(--border);
                overflow-x: auto;
            }
            .tab-btn {
                background: transparent;
                color: var(--text-muted);
                border: none;
                padding: 0.5rem 1rem;
                font-size: 0.95rem;
                font-weight: 500;
                cursor: pointer;
                border-bottom: 2px solid transparent;
                transition: all 0.3s ease;
                white-space: nowrap;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            .tab-btn.active {
                color: var(--primary);
                border-bottom-color: var(--primary);
            }
            .tab-btn:hover:not(.active) {
                color: var(--text-main);
            }
            .badge {
                background: var(--border);
                color: var(--text-main);
                padding: 0.1rem 0.5rem;
                border-radius: 12px;
                font-size: 0.75rem;
                font-weight: 700;
            }
            .tab-btn.active .badge {
                background: var(--primary);
                color: white;
            }
            .container {
                flex: 1;
                padding: 1.5rem 2rem;
                overflow-y: auto;
            }
            .orders-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
                gap: 1.5rem;
                align-items: start;
            }
            .card {
                background: var(--card-bg);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 1.5rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
                display: flex;
                flex-direction: column;
                position: relative;
                overflow: hidden;
            }
            .card.new-order {
                animation: highlight-pulse 2s ease-out;
            }
            @keyframes highlight-pulse {
                0% { box-shadow: 0 0 0 0 rgba(244, 63, 94, 0.7); border-color: var(--highlight); }
                70% { box-shadow: 0 0 0 15px rgba(244, 63, 94, 0); border-color: var(--border); }
                100% { box-shadow: 0 0 0 0 rgba(244, 63, 94, 0); border-color: var(--border); }
            }
            .card:hover {
                transform: translateY(-4px);
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
            }
            .card-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1rem;
                padding-bottom: 0.5rem;
                border-bottom: 1px solid rgba(255,255,255,0.05);
            }
            .phone-link {
                display: inline-flex;
                align-items: center;
                background: var(--whatsapp);
                color: white;
                text-decoration: none;
                padding: 0.4rem 0.8rem;
                border-radius: 20px;
                font-size: 0.85rem;
                font-weight: 600;
                transition: transform 0.2s ease;
            }
            .phone-link:hover { transform: scale(1.05); }
            .time { font-size: 0.75rem; color: var(--text-muted); }
            .info-group { margin-bottom: 0.8rem; }
            .info-label {
                font-size: 0.7rem;
                text-transform: uppercase;
                color: var(--text-muted);
                font-weight: 700;
                margin-bottom: 0.3rem;
                letter-spacing: 0.5px;
            }
            .info-value {
                font-size: 0.95rem;
                line-height: 1.4;
            }
            .total-box {
                background: rgba(16, 185, 129, 0.1);
                border: 1px solid rgba(16, 185, 129, 0.2);
                border-radius: 8px;
                padding: 0.8rem;
                margin: 1rem 0;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .total-label { color: var(--success); font-weight: 600; font-size: 0.9rem; }
            .total-amount { color: var(--success); font-weight: 700; font-size: 1.2rem; }
            .detail-box {
                background: rgba(0,0,0,0.2);
                padding: 0.8rem;
                border-radius: 6px;
                font-size: 0.85rem;
                white-space: pre-line;
                margin-top: 0.5rem;
                border-left: 3px solid var(--primary);
            }
            .actions {
                margin-top: 1rem;
                display: flex;
                gap: 0.5rem;
            }
            .btn {
                flex: 1;
                padding: 0.6rem;
                border: none;
                border-radius: 6px;
                font-weight: 600;
                cursor: pointer;
                transition: opacity 0.2s ease, transform 0.1s ease;
                color: white;
            }
            .btn:active { transform: scale(0.98); }
            .btn:hover { opacity: 0.9; }
            .btn-process { background: var(--primary); }
            .btn-dispatch { background: var(--success); }
            .btn-revert { background: var(--border); }
            
            .empty-state {
                grid-column: 1 / -1;
                text-align: center;
                padding: 4rem;
                color: var(--text-muted);
            }
            
            .live-duration {
                font-family: monospace;
                background: rgba(59, 130, 246, 0.2);
                color: var(--primary);
                padding: 0.2rem 0.5rem;
                border-radius: 4px;
                font-size: 0.85rem;
                font-weight: 700;
            }
            .timeline {
                margin-top: 1rem;
                padding-top: 1rem;
                border-top: 1px dashed var(--border);
                font-size: 0.8rem;
                color: var(--text-muted);
            }
            .timeline-item {
                display: flex;
                justify-content: space-between;
                margin-bottom: 0.3rem;
            }
            .timeline-status { font-weight: 600; color: var(--text-main); }
            
            /* Notificación flotante */
            #toast-container {
                position: fixed;
                bottom: 2rem;
                right: 2rem;
                display: flex;
                flex-direction: column;
                gap: 1rem;
                z-index: 1000;
            }
            .toast {
                background: var(--highlight);
                color: white;
                padding: 1rem 1.5rem;
                border-radius: 8px;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
                display: flex;
                align-items: center;
                gap: 1rem;
                transform: translateX(120%);
                animation: slide-in 0.3s forwards, slide-out 0.3s forwards 4s;
            }
            @keyframes slide-in { to { transform: translateX(0); } }
            @keyframes slide-out { to { transform: translateX(120%); opacity: 0; } }

            /* Responsive Design */
            @media (max-width: 768px) {
                header {
                    flex-direction: column;
                    align-items: flex-start;
                    gap: 1rem;
                    padding: 1rem;
                }
                .header-controls {
                    width: 100%;
                    justify-content: space-between;
                }
                .tabs {
                    padding: 0.8rem 1rem;
                }
                .container {
                    padding: 1rem;
                    padding-bottom: 6rem; /* Margen extra para que el último botón no quede tapado por la interfaz del teléfono */
                }
                .orders-grid {
                    grid-template-columns: 1fr;
                    gap: 1rem;
                }
                .card {
                    padding: 1rem;
                }
                .card-header {
                    flex-direction: column;
                    align-items: flex-start;
                    gap: 0.8rem;
                }
                .detail-box {
                    word-wrap: break-word;
                    word-break: break-word;
                }
            }
        </style>
    </head>
    <body>
        <header>
            <h1>Dashboard de Pedidos</h1>
            <div class="header-controls">
                <input type="date" id="filter-date" class="date-picker">
                <div id="last-updated" class="time">Actualizando...</div>
            </div>
        </header>
        <div class="tabs">
            <button class="tab-btn active" data-tab="Recientes">Recientes <span class="badge" id="badge-recientes">0</span></button>
            <button class="tab-btn" data-tab="En proceso">En proceso <span class="badge" id="badge-proceso">0</span></button>
            <button class="tab-btn" data-tab="Despachados">Despachados <span class="badge" id="badge-despachados">0</span></button>
        </div>
        <div class="container">
            <div id="orders-container" class="orders-grid">
                <!-- Orders injected here -->
            </div>
        </div>
        
        <div id="toast-container"></div>

        <script>
            let currentTab = 'Recientes';
            let allOrders = {};
            let knownRecentOrders = new Set();
            
            const dateFilter = document.getElementById('filter-date');
            
            // Set date picker to today in local timezone
            const today = new Date();
            const yyyy = today.getFullYear();
            const mm = String(today.getMonth() + 1).padStart(2, '0');
            const dd = String(today.getDate()).padStart(2, '0');
            dateFilter.value = `${yyyy}-${mm}-${dd}`;

            const tabs = document.querySelectorAll('.tab-btn');
            const container = document.getElementById('orders-container');
            const lastUpdated = document.getElementById('last-updated');

            tabs.forEach(tab => {
                tab.addEventListener('click', () => {
                    tabs.forEach(t => t.classList.remove('active'));
                    tab.classList.add('active');
                    currentTab = tab.dataset.tab;
                    renderOrders();
                });
            });
            
            dateFilter.addEventListener('change', () => {
                renderOrders();
            });

            function formatDuration(ms) {
                if (ms < 0) ms = 0;
                const totalSeconds = Math.floor(ms / 1000);
                const hours = Math.floor(totalSeconds / 3600);
                const minutes = Math.floor((totalSeconds % 3600) / 60);
                const seconds = totalSeconds % 60;
                
                let res = '';
                if (hours > 0) res += `${hours}h `;
                if (hours > 0 || minutes > 0) res += `${minutes}m `;
                res += `${seconds}s`;
                return res;
            }

            function updateLiveDurations() {
                const now = Date.now();
                document.querySelectorAll('.live-duration:not([data-static="true"])').forEach(el => {
                    const startTime = parseInt(el.dataset.time);
                    if (!isNaN(startTime)) {
                        el.innerText = '⏱️ ' + formatDuration(now - startTime);
                    }
                });
            }
            
            // Actualizar cronómetros cada segundo
            setInterval(updateLiveDurations, 1000);

            function showNotification(message) {
                const toastContainer = document.getElementById('toast-container');
                const toast = document.createElement('div');
                toast.className = 'toast';
                toast.innerHTML = `<span>🔔</span> <strong>Nuevo Pedido:</strong> ${message}`;
                toastContainer.appendChild(toast);
                
                // Play notification sound
                try {
                    const audio = new Audio('https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3');
                    audio.play().catch(e => console.log('Audio autoplay prevented'));
                } catch(e) {}

                setTimeout(() => {
                    if (toast.parentNode) toast.parentNode.removeChild(toast);
                }, 4500);
            }

            async function fetchOrders() {
                try {
                    const res = await fetch('/api/orders');
                    const newOrders = await res.json();
                    
                    // Check for new "Recientes" on TODAY's date to alert
                    const selectedDate = dateFilter.value;
                    let hasNewOrder = false;
                    
                    Object.values(newOrders).forEach(o => {
                        if (o.status === 'Recientes') {
                            const oDate = new Date(o.timestamp);
                            const oDateStr = `${oDate.getFullYear()}-${String(oDate.getMonth()+1).padStart(2,'0')}-${String(oDate.getDate()).padStart(2,'0')}`;
                            if (oDateStr === selectedDate) {
                                if (!knownRecentOrders.has(o.phone + o.timestamp)) {
                                    if (knownRecentOrders.size > 0) { // Don't alert on initial load
                                        hasNewOrder = true;
                                        showNotification(`${o.phone}`);
                                    }
                                    knownRecentOrders.add(o.phone + o.timestamp);
                                }
                            }
                        }
                    });

                    allOrders = newOrders;
                    renderOrders();
                    
                    const now = new Date();
                    lastUpdated.innerText = `Última act: ${now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit', hour12: true})}`;
                } catch (e) {
                    console.error('Error fetching orders:', e);
                }
            }

            async function changeStatus(orderId, newStatus) {
                try {
                    await fetch(`/api/orders/${orderId}/status`, {
                        method: 'PUT',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({status: newStatus})
                    });
                    // Refresh immediately
                    fetchOrders();
                } catch (e) {
                    console.error('Error changing status:', e);
                }
            }

            function renderOrders() {
                container.innerHTML = '';
                const selectedDateStr = dateFilter.value; // YYYY-MM-DD
                
                // Filter by Date
                let dayOrders = Object.values(allOrders).filter(o => {
                    const d = new Date(o.timestamp);
                    const orderDateStr = `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`;
                    return orderDateStr === selectedDateStr;
                });
                
                // Calculate Badges
                const counts = { 'Recientes': 0, 'En proceso': 0, 'Despachados': 0 };
                dayOrders.forEach(o => {
                    if (counts[o.status] !== undefined) counts[o.status]++;
                });
                
                document.getElementById('badge-recientes').innerText = counts['Recientes'];
                document.getElementById('badge-proceso').innerText = counts['En proceso'];
                document.getElementById('badge-despachados').innerText = counts['Despachados'];

                // Filter by Status Tab
                const tabOrders = dayOrders.filter(o => o.status === currentTab);
                
                // Sort by timestamp desc
                tabOrders.sort((a, b) => b.timestamp - a.timestamp);

                if (tabOrders.length === 0) {
                    container.innerHTML = `<div class="empty-state">
                        <h2 style="margin-bottom:0.5rem">No hay pedidos</h2>
                        <p>No se encontraron pedidos en esta sección para la fecha seleccionada.</p>
                    </div>`;
                    return;
                }

                tabOrders.forEach(order => {
                    const date = new Date(order.timestamp);
                    const timeStr = date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', hour12: true});
                    
                    const card = document.createElement('div');
                    card.className = 'card';
                    
                    // Highlight if it's very recent (last 30 seconds)
                    const isVeryRecent = (new Date().getTime() - order.timestamp) < 30000;
                    if (currentTab === 'Recientes' && isVeryRecent) {
                        card.classList.add('new-order');
                    }
                    
                    const orderId = order.id || order.phone; // Fallback para pedidos antiguos
                    let actionsHtml = '';
                    if (currentTab === 'Recientes') {
                        actionsHtml = `<button class="btn btn-process" onclick="changeStatus('${orderId}', 'En proceso')">Atender / En proceso</button>`;
                    } else if (currentTab === 'En proceso') {
                        actionsHtml = `<button class="btn btn-dispatch" onclick="changeStatus('${orderId}', 'Despachados')">Marcar Despachado</button>`;
                    } else {
                        actionsHtml = `<button class="btn btn-revert" onclick="changeStatus('${orderId}', 'Recientes')">Revertir a Recientes</button>`;
                    }

                    const detalle = order.detalle_completo ? `<div class="detail-box">${order.detalle_completo.replace(/\\n/g, '<br>')}</div>` : `<div class="info-value">${order.pedido || '-'}</div>`;
                    const totalBox = order.total_a_cobrar ? `
                        <div class="total-box">
                            <span class="total-label">Total a Cobrar</span>
                            <span class="total-amount">${order.total_a_cobrar}</span>
                        </div>
                    ` : '';

                    let timelineHtml = '';
                    if (order.history && order.history.length > 0) {
                        timelineHtml = '<div class="timeline"><div class="info-label">Traza de tiempos</div>';
                        for (let i = 0; i < order.history.length; i++) {
                            const step = order.history[i];
                            const stepDate = new Date(step.timestamp);
                            const stepTimeStr = stepDate.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit', hour12: true});
                            
                            let durationHtml = '';
                            if (i > 0) {
                                const diffMs = step.timestamp - order.history[i-1].timestamp;
                                durationHtml = `<span style="color:var(--warning)">(+${formatDuration(diffMs)})</span>`;
                            }
                            
                            timelineHtml += `
                                <div class="timeline-item">
                                    <span><span class="timeline-status">${step.status}</span> ${durationHtml}</span>
                                    <span>${stepTimeStr}</span>
                                </div>
                            `;
                        }
                        timelineHtml += '</div>';
                    }
                    let liveDurationHtml = `<div class="live-duration" data-time="${order.timestamp}">⏱️ 0s</div>`;
                    if (order.status === 'Despachados') {
                        let dispatchTime = order.timestamp;
                        if (order.history) {
                            const dispatchSteps = order.history.filter(h => h.status === 'Despachados');
                            if (dispatchSteps.length > 0) {
                                dispatchTime = dispatchSteps[dispatchSteps.length - 1].timestamp;
                            }
                        }
                        const totalTimeStr = formatDuration(dispatchTime - order.timestamp);
                        liveDurationHtml = `<div class="live-duration" style="background: rgba(16, 185, 129, 0.2); color: var(--success);" data-static="true">⏱️ ${totalTimeStr}</div>`;
                    }

                    card.innerHTML = `
                        <div class="card-header">
                            <a href="https://wa.me/${order.phone}" target="_blank" class="phone-link">
                                💬 ${order.phone}
                            </a>
                            <div style="display: flex; flex-direction: column; align-items: flex-end; gap: 0.3rem">
                                <span class="time">${timeStr}</span>
                                ${liveDurationHtml}
                            </div>
                        </div>
                        
                        <div class="info-group">
                            <div class="info-label">Detalle del Pedido</div>
                            ${detalle}
                        </div>
                        
                        ${totalBox}
                        
                        <div class="info-group">
                            <div class="info-label">Dirección de Entrega</div>
                            <div class="info-value">${order.direccion || '-'}</div>
                        </div>
                        
                        <div class="info-group">
                            <div class="info-label">Método de Pago</div>
                            <div class="info-value">${order.metodo_pago || '-'}</div>
                        </div>

                        <div class="info-group">
                            <div class="info-label">Resumen Adicional</div>
                            <div class="info-value" style="color: var(--text-muted); font-size: 0.8rem">${order.summary || '-'}</div>
                        </div>
                        
                        ${timelineHtml}
                        
                        <div class="actions">
                            ${actionsHtml}
                        </div>
                    `;
                    container.appendChild(card);
                });
                updateLiveDurations(); // Forzar actualización inicial para no mostrar 0s
            }

            // Initial fetch and interval
            fetchOrders();
            setInterval(fetchOrders, 10000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/panel")
async def serve_panel():
    return FileResponse("frontend/dist/index.html")


# Opcional: Montar assets
if os.path.exists("frontend/dist/assets"):
    app.mount(
        "/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")


@app.get("/favicon.ico")
async def favicon():
    return FileResponse("frontend/dist/favicon.svg")


@app.get("/{full_path:path}")
async def serve_spa(request: Request, full_path: str):
    if full_path.startswith("api/") or full_path.startswith("assets/") or full_path == "dashboard":
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse("frontend/dist/index.html")

if __name__ == "__main__":
    import uvicorn
    print(
        f"Modelos de fallback de Gemini configurados: {GEMINI_FALLBACK_MODELS_LIST}")
    print(
        f"Modelos compatibles con thinking_config: {THINKING_COMPATIBLE_MODELS}")
    print("Iniciando servidor FastAPI en http://0.0.0.0:8000")
    print("Swagger UI disponible en http://0.0.0.0:8000/docs")
    print("Versión: 1.2.32_gemini_thinking_config")
    print("Cambios principales:")
    print("- Delay entre mensajes cambiado de 1.5s a 0s")
    print("- Logs mejorados para mostrar TODOS los modelos intentados")
    print("- Agregado soporte para thinking_config en modelos Gemini 2.5")
    print("- Nuevo parámetro thinking_budget: -1 (razonar) o 0 (no razonar)")

    port = int(os.getenv("PORT", 8000))
    print(f"Puerto configurado: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
