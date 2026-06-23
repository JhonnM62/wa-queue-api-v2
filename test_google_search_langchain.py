import asyncio
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

async def test_search():
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.0-flash", 
        temperature=0, 
        google_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    )
    
    tools = [{"google_search": {}}]
    
    try:
        llm_with_tools = llm.bind_tools(tools)
        response = await llm_with_tools.ainvoke("A qué hora es el partido de Colombia hoy?")
        print("Response:", response)
        print("Tool calls:", response.tool_calls)
    except Exception as e:
        print("Error:", e)

asyncio.run(test_search())
