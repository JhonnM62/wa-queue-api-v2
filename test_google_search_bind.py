import os
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Test how to bind google_search
try:
    llm_with_tools = llm.bind_tools(["google_search"])
    res = llm_with_tools.invoke("When is the next match for Colombia?")
    print("Success with string 'google_search':", res.content)
except Exception as e:
    print("Failed with string 'google_search':", e)

try:
    from google.genai import types
    tool = types.Tool(google_search=types.GoogleSearch())
    llm_with_tools = llm.bind_tools([tool])
    res = llm_with_tools.invoke("When is the next match for Colombia?")
    print("Success with types.Tool:", res.content)
except Exception as e:
    print("Failed with types.Tool:", e)
