import os
import ssl

import urllib3

# 1. Parche global para ignorar certificados autofirmados en red corporativa
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

old_init = urllib3.connection.HTTPSConnection.__init__


def new_init(self, *args, **kwargs):
    kwargs["cert_reqs"] = "CERT_NONE"
    old_init(self, *args, **kwargs)


urllib3.connection.HTTPSConnection.__init__ = new_init

# 2. Demás importaciones ordenadas alfabéticamente
import httpx
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

DATA_PATH = "./data"


def indexar_documentos_en_nube():
    api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("INDEX_NAME", "rag-tecnico-index")

    if not api_key or not pinecone_api_key:
        raise ValueError("Faltan credenciales de OpenAI o Pinecone en las variables de entorno.")

    # 3. Inicialización de Pinecone
    pc = Pinecone(api_key=pinecone_api_key)

    try:
        existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ Aviso de conexión: {e}")
        existing_indexes = []

    if index_name not in existing_indexes:
        print(f"☁️ Creando índice Serverless '{index_name}' en Pinecone...")
        try:
            pc.create_index(
                name=index_name,
                dimension=1536,  # Dimensión para text-embedding-3-small
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            print("✅ ¡Índice creado con éxito!")
        except Exception as ex:  # noqa: BLE001
            print(f"ℹ️ Nota sobre el índice (puede que ya exista): {ex}")
    else:
        print(f"ℹ️ El índice '{index_name}' ya existe en la nube.")

    # 4. Carga de documentos
    print("📁 Cargando documentos desde la carpeta /data...")
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()

    if not docs:
        print("❌ No se encontraron documentos en la carpeta /data.")
        return

    # 5. Chunking estratégico
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_documents(docs)
    print(f"🧩 Se generaron {len(chunks)} fragmentos (chunks).")

    # 6. Embeddings e Ingesta en Pinecone
    http_client = httpx.Client(verify=False)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
        http_client=http_client,
    )

    print("🚀 Subiendo vectores y metadatos a Pinecone Serverless...")
    PineconeVectorStore.from_documents(documents=chunks, embedding=embeddings, index_name=index_name)
    print("✅ Ingesta en la nube completada con éxito.")


if __name__ == "__main__":
    indexar_documentos_en_nube()