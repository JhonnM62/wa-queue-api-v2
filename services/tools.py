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
    from datetime import datetime

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
    ORDERS_FILE_PATH = "orders.json"
    order_timestamp = datetime.now().isoformat()
    order_id = f"ORDER_{int(time.time())}_{user_phone[-4:]}"

    try:
        orders_data = {}
        if os.path.exists(ORDERS_FILE_PATH):
            with open(ORDERS_FILE_PATH, 'r', encoding='utf-8') as f:
                orders_data = json.load(f)
                
        orders_data[order_id] = {
            "phone": user_phone,
            "summary": resumen_general,
            "pedido": lista_productos,
            "detalle_completo": f"Resumen: {resumen_general}\nProductos: {lista_productos}\nTotal: {total_a_cobrar}\nDirección: {direccion_envio}\nPago: {metodo_pago}",
            "total_a_cobrar": total_a_cobrar,
            "direccion": direccion_envio,
            "metodo_pago": metodo_pago,
            "status": "Recientes",
            "timestamp": order_timestamp,
            "static_duration": 0,
            "history": [{"status": "Recientes", "timestamp": order_timestamp}],
            "origen": "tool_langgraph"
        }

        with open(ORDERS_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(orders_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error al guardar en dashboard: {e}")

    # 2. Enviar notificación HTTP
    send_url = f"{server_url}/chats/send?id={userbot}"
    headers = {"Content-Type": "application/json", "x-access-token": token}
    
    link_message_text = f"https://wa.me/{user_phone}"
    detail_message_text = f"Nuevo Pedido (vía LangGraph Tool):\n\nProductos: {lista_productos}\nTotal: {total_a_cobrar}\nDir: {direccion_envio}\nPago: {metodo_pago}"
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
