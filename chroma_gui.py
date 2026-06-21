import os
from flask import Flask, render_template_string
import chromadb

app = Flask(__name__)

CHROMA_PERSIST_DIR = "./chroma_db"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ChromaDB Viewer</title>
    <style>
        body { font-family: sans-serif; background: #1e1e2f; color: #fff; padding: 20px; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 10px; border: 1px solid #444; text-align: left; }
        th { background: #2d2d44; }
        .content { max-height: 150px; overflow-y: auto; white-space: pre-wrap; font-size: 13px; }
        .meta { color: #aaa; font-size: 12px; }
    </style>
</head>
<body>
    <h2>ChromaDB Viewer (wa_knowledge_base)</h2>
    <p>Documentos totales en la colección: {{ total }}</p>
    <table>
        <tr>
            <th>ID</th>
            <th>Bot ID (Metadata)</th>
            <th>Source (Metadata)</th>
            <th>Content</th>
        </tr>
        {% for doc in documents %}
        <tr>
            <td>{{ doc.id }}</td>
            <td><span class="meta">{{ doc.bot_id }}</span></td>
            <td><span class="meta">{{ doc.source }}</span></td>
            <td><div class="content">{{ doc.content }}</div></td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
"""

@app.route("/")
def index():
    if not os.path.exists(CHROMA_PERSIST_DIR):
        return "El directorio chroma_db no existe aún."
    
    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        collection = client.get_collection(name="wa_knowledge_base")
        results = collection.get()
        
        documents = []
        for i in range(len(results['ids'])):
            documents.append({
                "id": results['ids'][i],
                "content": results['documents'][i] if results['documents'] else "",
                "bot_id": results['metadatas'][i].get("bot_id", "N/A") if results['metadatas'] else "N/A",
                "source": results['metadatas'][i].get("source", "N/A") if results['metadatas'] else "N/A"
            })
            
        return render_template_string(HTML_TEMPLATE, documents=documents, total=len(documents))
    except Exception as e:
        return f"Error al leer ChromaDB: {e}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081)
