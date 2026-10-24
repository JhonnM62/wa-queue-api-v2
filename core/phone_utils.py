import re
from typing import Optional


def is_meta_identifier(identifier: Optional[str]) -> bool:
    """
    Determina si un identificador corresponde a un ID interno de Meta,
    ManyChat, WhatsApp LID o nombre de usuario (@username) en lugar de un teléfono celular real.
    
    Casos comunes:
    - Usernames: '@carlos_restrepo', 'user_12345'
    - LIDs de WhatsApp: '140222780145809@lid', '179847494119431'
    - Meta Page/User Scoped IDs: 14 a 18 dígitos numéricos
    - Cuentas con privacidad/número oculto en WhatsApp o ManyChat
    """
    if not identifier:
        return False
    raw = str(identifier).strip()

    # 1. Si contiene @ (ej. @usuario, @lid) o letras o guión bajo
    if "@" in raw or any(c.isalpha() for c in raw) or "_" in raw:
        return True

    # 2. Dejar solo dígitos
    digits = re.sub(r'\D', '', raw)
    if not digits:
        return True

    # 3. LIDs y Meta Scoped IDs: típicamente tienen 14 o más dígitos (ej. 140222780145809, 179847494119431)
    if len(digits) >= 14:
        return True

    # 4. Longitud muy corta para ser teléfono válido
    if len(digits) < 10:
        return True

    # 5. Validación de números reales (Colombia y frontera Ecuador):
    # - Móvil Colombia: 10 dígitos (3XXXXXXXXX) o 12 dígitos (573XXXXXXXXX)
    # - Fijo Colombia: 10 dígitos (60XXXXXXXX) o 12 dígitos (5760XXXXXXX)
    # - Móvil Ecuador: 12 dígitos (5939XXXXXXXX) o 11 dígitos
    if len(digits) == 10 and not (digits.startswith("3") or digits.startswith("60")):
        return True

    if len(digits) == 12 and not (digits.startswith("573") or digits.startswith("5760") or digits.startswith("593")):
        return True

    if len(digits) == 13:
        return True

    if len(digits) == 11 and not digits.startswith("593"):
        return True

    return False


def clean_real_phone(raw_phone: Optional[str]) -> str:
    """
    Limpia y estandariza un número celular real.
    Busca un número móvil colombiano (10 dígitos que empiezan por 3 o 12 dígitos que empiezan por 573).
    Si no corresponde a un teléfono móvil real (o es un ID de Meta / LID / texto), retorna cadena vacía "".
    """
    if not raw_phone:
        return ""
    raw = str(raw_phone).strip()

    # Si contiene un número móvil colombiano estándar de 10 o 12 dígitos
    match_57 = re.search(r'\b57(3\d{9})\b', raw)
    if match_57:
        return "57" + match_57.group(1)

    match_10 = re.search(r'\b(3\d{9})\b', raw)
    if match_10:
        return "57" + match_10.group(1)

    digits = re.sub(r'\D', '', raw)
    if len(digits) == 10 and digits.startswith("3"):
        return "57" + digits
    if len(digits) == 12 and digits.startswith("573"):
        return digits
    if len(digits) == 12 and digits.startswith("593"):  # Soporte frontera Ecuador
        return digits

    return ""


def extract_provided_phone(raw_phone: Optional[str]) -> str:
    """
    Extrae y limpia el número telefónico proporcionado por el cliente en el chat.
    Si el formato es estándar colombiano/ecuatoriano, lo normaliza con clean_real_phone.
    Si el cliente lo digitó con error leve (menos o más dígitos, sin código de país, etc.),
    confía en lo que digitó extrayendo los dígitos limpios en lugar de descartarlo.
    Retorna "" si está vacío o no contiene al menos 7 dígitos.
    """
    if not raw_phone:
        return ""
    raw = str(raw_phone).strip()
    if raw.lower() in ["", "no especificado", "no aplica", "n/a", "na", "ninguno", "null", "none"]:
        return ""

    # 1. Intentar normalización estándar (ej. 3123456789 -> 573123456789)
    cleaned = clean_real_phone(raw)
    if cleaned:
        return cleaned

    # 2. Confiamos en el número que el cliente proporcionó:
    digits = re.sub(r'\D', '', raw)
    if len(digits) >= 7:
        if len(digits) == 10 and digits.startswith("3"):
            return "57" + digits
        return digits

    return ""


def resolve_order_phones(
    api_linea_wa: Optional[str],
    provided_phone: Optional[str],
    nombre_cliente: Optional[str] = None
) -> dict:
    """
    Resuelve la asignación del teléfono y presentación según la regla de negocio:

    1. Contactos SIN Meta ID (vienen desde un número normal de WhatsApp):
       - Se toma SIEMPRE el número que viene desde la API (api_linea_wa).
       - NO se sobreescribe con lo que el cliente haya escrito en el chat para evitar
         errores si el cliente lo digitó mal.
       - assigned_phone = api_linea_wa
       - display_phone = api_linea_wa
       - meta_identifier = None
       - extracted_phone = lo que escribió el cliente (solo para registro/auditoría)
       - has_real_phone = True

    2. Contactos CON Meta ID (LID, ManyChat, @username, etc.):
       - No existe número celular en la API (api_linea_wa es el identificador de Meta).
       - Se toma el número que proporciona el cliente (provided_phone).
       - Se confía en el número proporcionado (esté bien o mal digitado).
       - Si el cliente dio un número:
           assigned_phone = provided_phone limpio
           display_phone = provided_phone limpio
           meta_identifier = api_linea_wa
           has_real_phone = True
       - Si el cliente NO dio número:
           assigned_phone = api_linea_wa (fallback)
           display_phone = api_linea_wa
           meta_identifier = api_linea_wa
           has_real_phone = False
    """
    raw_api = str(api_linea_wa or "").strip()
    is_meta = is_meta_identifier(raw_api)
    extracted_clean = extract_provided_phone(provided_phone) if provided_phone else ""

    nombre_clean = ""
    if nombre_cliente and str(nombre_cliente).strip().lower() not in ["", "no especificado", "no aplica", "n/a", "na", "null", "none"]:
        nombre_clean = str(nombre_cliente).strip()

    if is_meta:
        # Contacto CON Meta ID:
        # Se toma el número proporcionado por el cliente
        if extracted_clean:
            assigned_phone = extracted_clean
            display_phone = extracted_clean
            has_real_phone = True
            if nombre_clean:
                cliente_str = f"{nombre_clean} ({display_phone}) [Meta ID: {raw_api}]"
            else:
                cliente_str = f"{display_phone} [Meta ID: {raw_api}]"
            link_message_text = f"https://wa.me/{display_phone}"
        else:
            # No proporcionó número real aún
            assigned_phone = raw_api
            display_phone = raw_api
            has_real_phone = False
            if nombre_clean:
                cliente_str = f"{nombre_clean} ⚠️ [Sin número real] (Meta ID: {raw_api})"
            else:
                cliente_str = f"⚠️ [Sin número real] Meta ID: {raw_api}"
            link_message_text = (
                f"⚠️ Contacto Meta / Anónimo: {raw_api}\n"
                f"🚨 ATENCIÓN: El cliente NO proporcionó un número celular real. Se requiere contactarlo por chat de la plataforma."
            )

        meta_identifier = raw_api
    else:
        # Contacto SIN Meta ID:
        # Se toma SIEMPRE el número que viene desde la API.
        # No se le asigna lo que digitó el cliente para evitar errores si lo escribió mal.
        assigned_phone = raw_api
        display_phone = raw_api
        has_real_phone = True
        meta_identifier = None
        if nombre_clean:
            cliente_str = f"{nombre_clean} ({display_phone})"
        else:
            cliente_str = display_phone
        link_message_text = f"https://wa.me/{display_phone}"

    return {
        "is_meta": is_meta,
        "assigned_phone": assigned_phone,
        "display_phone": display_phone,
        "meta_identifier": meta_identifier,
        "extracted_phone": extracted_clean or (str(provided_phone).strip() if provided_phone else None),
        "has_real_phone": has_real_phone,
        "cliente_str": cliente_str,
        "link_message_text": link_message_text
    }


def get_meta_directive_prompt(identifier: str) -> str:
    """
    Genera la directiva prioritaria e ineludible para el prompt del agente cuando
    el cliente tiene un identificador de Meta / ManyChat / Username en vez de un teléfono real.
    """
    return (
        "\n\n============================================================\n"
        "REGLA ESTRICTA E INELUDIBLE: CONTACTO CON IDENTIFICADOR DE META / NUMERO OCULTO\n"
        f"El remitente actual está conectado mediante un ID de Meta o nombre de usuario ('{identifier}') y NO muestra su número telefónico real.\n\n"
        "INSTRUCCION OBLIGATORIA:\n"
        "1. DEBES EXIGIRLE AL CLIENTE QUE TE SUMINISTRE SU NUMERO DE TELEFONO CELULAR REAL (ejemplo: 3101234567 o 3XXXXXXXXX).\n"
        "   Explícale amablemente pero con firmeza que los domiciliarios y el restaurante necesitan llamarlo o contactarlo por WhatsApp para la entrega de su pedido.\n"
        "2. PROHIBICION ABSOLUTA DE CIERRE: NO uses el estado '2. Flujo de venta exitoso' NI confirmes el pedido definitivamente mientras el cliente NO haya escrito su número de teléfono celular real en la conversación.\n"
        "3. Si el cliente ya dio sus productos y dirección pero NO ha dado su número celular real, mantén el estado en '1. Esperando confirmación del cliente' y pídele su número celular en tu mensaje.\n"
        "4. En la herramienta `enviar_notificacion_tool`, envía en el campo `telefono_cliente` el número celular real que el cliente escribió.\n"
        "============================================================\n"
    )
