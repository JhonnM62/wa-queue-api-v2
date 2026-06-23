import sqlite3
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

db = sqlite3.connect("saas_data.db")
c = db.cursor()
c.execute("SELECT apikey FROM bots LIMIT 1")
row = c.fetchone()
api_key = row[0] if row else None
db.close()

@tool
def buscar_catalogo(query: str):
    """Busca en el catálogo."""
    return f"Resultado del catálogo para {query}"

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=api_key)

try:
    gemini_tool_config = {
        "function_calling_config": {"mode": "AUTO"},
        "include_server_side_tool_invocations": True
    }
    llm_with_tools = llm.bind_tools([{"google_search": {}}, buscar_catalogo], tool_config=gemini_tool_config)
    res = llm_with_tools.invoke("Busca zapatos y el partido de colombia")
    print("Success:", res.content)
except Exception as e:
    print("Failed with tool_config:", e)

try:
    # Another way?
    llm_with_tools = llm.bind_tools([{"google_search": {}}, buscar_catalogo], **{"tool_config": gemini_tool_config})
    res = llm_with_tools.invoke("Busca zapatos y el partido de colombia")
    print("Success with kwargs:", res.content)
except Exception as e:
    print("Failed with kwargs:", e)
