import os
import shutil
from fastapi import UploadFile

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
import google.generativeai as genai

CHROMA_PERSIST_DIR = "./chroma_db"
TEMP_UPLOAD_DIR = "./temp_uploads"

os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)


def _get_embeddings(api_key: str) -> GoogleGenerativeAIEmbeddings:
    """
    Crea una instancia de embeddings usando la api_key del bot/cliente.
    No usa variables de entorno — la clave viene siempre del BotConfig
    (que a su vez se rellena desde el campo `apikey` de la petición de WhatsApp).
    """
    if not api_key:
        raise ValueError(
            "Este bot no tiene una API key configurada. "
            "Envía al menos una petición de WhatsApp para que se registre automáticamente, "
            "o agrégala manualmente en el panel."
        )
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )


def _get_vector_store(api_key: str) -> Chroma:
    return Chroma(
        collection_name="wa_knowledge_base",
        embedding_function=_get_embeddings(api_key),
        persist_directory=CHROMA_PERSIST_DIR
    )


async def process_and_store_document(bot_id: int, file: UploadFile, api_key: str) -> str:
    """
    Procesa un PDF o TXT y lo guarda en ChromaDB con el bot_id como metadato.
    Usa la api_key del cliente (almacenada en BotConfig) o del entorno (.env) para los embeddings.
    """
    actual_api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not actual_api_key:
        raise ValueError("No se encontró una API Key válida para Gemini. Configúrala en el bot o en .env")

    file_path = os.path.join(TEMP_UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    documents = []
    try:
        if file.filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        elif file.filename.lower().endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
        elif file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            genai.configure(api_key=actual_api_key)
            model = genai.GenerativeModel("gemini-3.5-flash")
            
            prompt = (
                "Extrae toda la información de este catálogo/menú. "
                "Lista claramente: 1. Los productos 2. Sus características o ingredientes 3. Sus precios. "
                "Formatea la respuesta como texto claro y estructurado."
            )
            
            import PIL.Image
            with PIL.Image.open(file_path) as img:
                response = model.generate_content([prompt, img])
            
            extracted_text = response.text
            if not extracted_text:
                raise ValueError("No se pudo extraer texto de la imagen.")
                
            documents = [Document(page_content=extracted_text, metadata={"source": file.filename})]
        else:
            raise ValueError("Formato no soportado. Usa PDF, TXT o imágenes (PNG/JPG/WEBP).")

        for doc in documents:
            doc.metadata["bot_id"] = bot_id
            doc.metadata["source"] = file.filename

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        vector_store = _get_vector_store(actual_api_key)
        vector_store.add_documents(chunks)

        return f"Procesados {len(chunks)} fragmentos de '{file.filename}'."

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


def get_all_knowledge(bot_id: int, api_key: str):
    """
    Recupera todos los fragmentos de conocimiento guardados para un bot.
    """
    actual_api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not actual_api_key:
        return []
        
    try:
        vector_store = _get_vector_store(actual_api_key)
        # We access the underlying collection to get documents by metadata
        collection = vector_store._collection
        results = collection.get(where={"bot_id": bot_id})
        
        knowledge = []
        for i in range(len(results['ids'])):
            knowledge.append({
                "id": results['ids'][i],
                "content": results['documents'][i],
                "source": results['metadatas'][i].get("source", "Desconocido")
            })
        return knowledge
    except Exception as e:
        import traceback
        traceback.print_exc()
        return []


def delete_bot_knowledge(bot_id: int, api_key: str):
    """
    Elimina todo el conocimiento RAG asociado a un bot_id en ChromaDB.
    """
    actual_api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not actual_api_key:
        return
        
    try:
        vector_store = _get_vector_store(actual_api_key)
        collection = vector_store._collection
        
        # Elimina usando where filter en los metadatos
        collection.delete(where={"bot_id": bot_id})
        print(f"✅ Conocimiento RAG eliminado exitosamente para el bot {bot_id}")
    except Exception as e:
        print(f"Error al eliminar conocimiento RAG para el bot {bot_id}: {e}")
        import traceback
        traceback.print_exc()


def search_knowledge(bot_id: int, query: str, api_key: str, top_k: int = 3) -> str:
    """
    Busca en ChromaDB filtrando por bot_id.
    Requiere la api_key del cliente para inicializar los embeddings.
    """
    vector_store = _get_vector_store(api_key)
    results = vector_store.similarity_search(
        query,
        k=top_k,
        filter={"bot_id": bot_id}
    )

    if not results:
        return "No se encontró información relevante en el catálogo/documentos."

    return "\n\n---\n\n".join([doc.page_content for doc in results])


def update_single_knowledge(bot_id: int, doc_id: str, new_content: str, api_key: str):
    """
    Actualiza el contenido de un fragmento específico en ChromaDB y su embedding.
    """
    actual_api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not actual_api_key:
        raise ValueError("API Key no configurada para este bot")
        
    vector_store = _get_vector_store(actual_api_key)
    collection = vector_store._collection
    
    # Verificar que el documento pertenece a este bot
    result = collection.get(ids=[doc_id])
    if not result or not result['ids']:
        raise ValueError("Documento no encontrado")
        
    metadata = result['metadatas'][0]
    if metadata.get('bot_id') != bot_id:
        raise ValueError("El documento no pertenece a este bot")
        
    # Actualizar documento usando langchain para asegurar que se regenere el embedding
    from langchain_core.documents import Document
    new_doc = Document(page_content=new_content, metadata=metadata)
    vector_store.update_document(doc_id, new_doc)

def delete_single_knowledge(bot_id: int, doc_id: str, api_key: str):
    """
    Elimina un fragmento específico en ChromaDB.
    """
    actual_api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not actual_api_key:
        raise ValueError("API Key no configurada para este bot")
        
    vector_store = _get_vector_store(actual_api_key)
    collection = vector_store._collection
    
    # Verificar que el documento pertenece a este bot
    result = collection.get(ids=[doc_id])
    if not result or not result['ids']:
        raise ValueError("Documento no encontrado")
        
    metadata = result['metadatas'][0]
    if metadata.get('bot_id') != bot_id:
        raise ValueError("El documento no pertenece a este bot")
        
    # Eliminar documento
    collection.delete(ids=[doc_id])
