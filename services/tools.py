from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from services.rag_service import search_knowledge


@tool
def buscar_catalogo_tool(query: str, config: RunnableConfig) -> str:
    """Busca información en el catálogo o base de conocimiento del chatbot.
    Úsala SIEMPRE que el usuario pregunte por precios, productos, servicios, horarios o dudas de negocio.
    """
    bot_id = config["configurable"].get("bot_id")
    api_key = config["configurable"].get("config_dict", {}).get("api_key")
    return search_knowledge(bot_id=bot_id, query=query, api_key=api_key)


@tool
def enviar_notificacion_tool(
    resumen_general: str, 
    lista_productos: str, 
    total_a_cobrar: str, 
    direccion_envio: str, 
    metodo_pago: str, 
    config: RunnableConfig
) -> str:
    """Envía una notificación al vendedor y guarda el pedido en el dashboard de ventas.
    Úsala SOLO cuando se haya concretado un pedido final.
    Debes extraer de la conversación el resumen general, la lista de productos, el total a cobrar, la dirección y el método de pago.
    Si algún dato no existe o no aplica, pasa una cadena vacía o "N/A".
    """
    import httpx
    import json
    import os
    import time
    from datetime import datetime, timezone

    config_dict = config["configurable"].get("config_dict", {})
    
    server_url = config_dict.get("server")
    token = config_dict.get("token")
    receiver = config_dict.get("receiver")
    is_group = config_dict.get("is_group", False)
    userbot = config_dict.get("userbot_identifier")
    user_phone = config_dict.get("user_phone", "")

    if not server_url or not receiver:
        return "Fallo: No se encontraron los datos del servidor o del receptor para notificar."

    # 1. Guardar en el dashboard (orders.json)
    ORDERS_FILE_PATH = "./Download/AutoSystem/orders.json"
    order_timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)

    # Formatear lista_productos con emoji 👉 para consistencia visual con WhatsApp
    def format_detalle_wa(texto: str) -> str:
        if not texto:
            return texto
        lineas = texto.strip().split('\n')
        resultado = []
        for linea in lineas:
            linea_strip = linea.strip()
            if not linea_strip:
                resultado.append('')
                continue
            # Agregar 👉 a líneas de productos (no a sub-ítems con guión ni a líneas ya formateadas)
            if (not linea_strip.startswith('👉') and
                    not linea_strip.startswith('-') and
                    not linea_strip.startswith('•')):
                linea_strip = f"👉 {linea_strip}"
            resultado.append(linea_strip)
        return '\n'.join(resultado)

    detalle_completo = format_detalle_wa(lista_productos)

    # --- Lógica de deduplicación unificada con main.py ---
    # Usar el teléfono como order_id base para evitar duplicados entre las dos rutas
    is_modification = False
    existing_order_id = None

    try:
        orders_data = {}
        if os.path.exists(ORDERS_FILE_PATH):
            with open(ORDERS_FILE_PATH, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    orders_data = json.loads(content)

        # Buscar pedido activo del mismo teléfono en el día de hoy (cualquier status excepto Despachados)
        today = datetime.now(timezone.utc).date()
        for oid, odata in orders_data.items():
            if odata.get("phone") == user_phone and odata.get("status") != "Despachados":
                order_ts = odata.get("timestamp", 0)
                if order_ts:
                    order_date = datetime.fromtimestamp(order_ts / 1000, tz=timezone.utc).date()
                    if order_date == today:
                        is_modification = True
                        existing_order_id = oid
                        break

        if is_modification and existing_order_id:
            # ACTUALIZAR in-place: preservar timestamp original, status e historial
            orders_data[existing_order_id]["summary"] = resumen_general
            
            # Evitar que la IA sobrescriba el detalle con un resumen genérico al modificar el pago
            # Detectamos frases como "Productos por un total de" o "(Detalle en historial)"
            texto_lower = lista_productos.lower()
            is_generic_summary = "historial" in texto_lower or "productos por un total" in texto_lower
            
            if not is_generic_summary:
                orders_data[existing_order_id]["pedido"] = lista_productos
                orders_data[existing_order_id]["detalle_completo"] = detalle_completo
                
            orders_data[existing_order_id]["total_a_cobrar"] = total_a_cobrar
            orders_data[existing_order_id]["direccion"] = direccion_envio
            orders_data[existing_order_id]["metodo_pago"] = metodo_pago
            order_id = existing_order_id
            
            # Si preservamos el detalle anterior, asegurarnos de usarlo para la notificación HTTP de WhatsApp
            if is_generic_summary:
                detalle_completo = orders_data[existing_order_id].get("detalle_completo", detalle_completo)
                
            print(f"[tools/enviar_notificacion/{user_phone}] Pedido actualizado en dashboard (ID: {order_id}). Sin duplicado.")
        else:
            # Pedido nuevo — usar prefijo ORDER_ original
            order_id = f"ORDER_{int(time.time())}_{user_phone[-4:]}"

            orders_data[order_id] = {
                "id": order_id,
                "phone": user_phone,
                "summary": resumen_general,
                "pedido": lista_productos,
                "detalle_completo": detalle_completo,
                "total_a_cobrar": total_a_cobrar,
                "direccion": direccion_envio,
                "metodo_pago": metodo_pago,
                "status": "Recientes",
                "timestamp": order_timestamp,
                "static_duration": 0,
                "history": [{"status": "Recientes", "timestamp": order_timestamp}],
                "origen": "tool_langgraph"
            }
            print(f"[tools/enviar_notificacion/{user_phone}] Nuevo pedido guardado en dashboard (ID: {order_id}).")

        with open(ORDERS_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(orders_data, f, ensure_ascii=False, indent=4)

    except Exception as e:
        print(f"[tools/enviar_notificacion/{user_phone}] Error al guardar en dashboard: {e}")

    # 2. Enviar notificación HTTP a WhatsApp
    send_url = f"{server_url}/chats/send?id={userbot}"
    headers = {"Content-Type": "application/json", "x-access-token": token}
    
    link_message_text = f"https://wa.me/{user_phone}"
    
    header_title = "🛍️ *NUEVO PEDIDO* 🛍️"
    if is_modification:
        header_title = "🔄 *PEDIDO MODIFICADO* 🔄"

    detail_message_text = (
        f"{header_title}\n\n"
        f"👤 *Cliente:* {user_phone}\n\n"
        f"📝 *Detalle del Pedido:*\n{detalle_completo}\n\n"
        f"💵 *Total a Cobrar:* {total_a_cobrar}\n"
        f"💳 *Método de Pago:* {metodo_pago}\n"
        f"📍 *Dirección de Entrega:* {direccion_envio}\n\n"
        f"🤖 *Resumen IA:* {resumen_general}"
    )
    combined_message_text = f"{detail_message_text}\n\n{link_message_text}"

    combined_payload = {
        "receiver": receiver,
        "isGroup": is_group,
        "message": {
            "text": combined_message_text
        }
    }

    try:
        # Petición síncrona ya que estamos dentro de una ejecución de herramienta síncrona
        resp = httpx.post(send_url, headers=headers, json=combined_payload, timeout=30.0)
        resp.raise_for_status()
        return "Notificación enviada y pedido registrado en dashboard exitosamente."
    except Exception as e:
        return f"Error al enviar la notificación HTTP: {e}"
