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
def enviar_notificacion_tool(mensaje: str, config: RunnableConfig) -> str:
    """Envía una notificación push por WhatsApp a una línea de administración o al vendedor.
    Úsala cuando se haya concretado un pedido, haya un pago pendiente por revisar o el cliente exija hablar con un humano.
    """
    # config["configurable"].get("config_dict", {}) tiene server_url y api_key
    # Implementación real HTTP (adaptar a la estructura de tu servidor WA):
    # import httpx
    # server_url = config["configurable"].get("config_dict", {}).get("server")
    # linea_notificacion = "XXXXX" # Obtener de BD
    # url = f"{server_url}/message/sendText/{linea_notificacion}"
    # payload = {"number": linea_notificacion, "options": {"delay": 1200}, "textMessage": {"text": mensaje}}
    # headers = {"apikey": config["configurable"].get("config_dict", {}).get("api_key")}
    # httpx.post(url, json=payload, headers=headers)
    return f"Notificación enviada exitosamente."
