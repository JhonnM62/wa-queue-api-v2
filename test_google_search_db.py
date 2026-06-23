import sqlite3
import json

db = sqlite3.connect("saas_data.db")
c = db.cursor()
c.execute("SELECT apikey FROM bots LIMIT 1")
row = c.fetchone()
api_key = row[0] if row else None
db.close()

if not api_key:
    print("No API key found in DB.")
    exit(1)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

@tool
def buscar_catalogo(query: str):
    """Busca en el catálogo."""
    return f"Resultado del catálogo para {query}"

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=api_key)

try:
    print("Binding google_search tool and custom tool...")
    llm_with_tools = llm.bind_tools([{"google_search": {}}, buscar_catalogo])
    print("Invoking...")
    res = llm_with_tools.invoke("Busca en el catálogo la palabra 'zapatos' y dime cuando juega Colombia")
    print("Response Content:")
    print(res.content)
    print("Tool Calls:")
    print(res.tool_calls)
except Exception as e:
    print("Failed:", e)
