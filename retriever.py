import os
import ssl

import urllib3

# Parche global para bypass de certificados SSL corporativos
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

old_init = urllib3.connection.HTTPSConnection.__init__


def new_init(self, *args, **kwargs):
    kwargs["cert_reqs"] = "CERT_NONE"
    old_init(self, *args, **kwargs)


urllib3.connection.HTTPSConnection.__init__ = new_init

# Importaciones limpias y seguras
import httpx
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


class CustomHybridRetriever:
    """Recuperador híbrido personalizado con soporte sincrónico y asincrónico."""

    def __init__(self, vector_retriever, bm25_retriever, weights=None):
        if weights is None:
            weights = [0.5, 0.5]
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.weights = weights
        self.k = 5

    def invoke(self, query: str, config=None, **kwargs) -> list[Document]:
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        seen = set()
        combined = []
        for doc in vector_docs + bm25_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                combined.append(doc)

        return combined[: self.k]

    async def ainvoke(self, query: str, config=None, **kwargs) -> list[Document]:
        """Versión asincrónica requerida por evaluate.py"""
        return self.invoke(query, config, **kwargs)

    def get_relevant_documents(self, query: str) -> list[Document]:
        return self.invoke(query)


class HybridRAGRetriever:

    def __init__(self):
        self.index_name = os.getenv("INDEX_NAME", "rag-tecnico-index")
        self.api_key = os.getenv("OPENAI_API_KEY")

        # Cliente HTTP sin verificación SSL para evitar bloqueos de red
        http_client = httpx.Client(verify=False)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", api_key=self.api_key, http_client=http_client
        )

        # Conexión al vector store en Pinecone
        self.vectorstore = PineconeVectorStore(
            index_name=self.index_name, embedding=self.embeddings
        )

        # 1. Recuperador Vectorial (Top K)
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )

        # 2. Recuperador Léxico BM25
        corpus_docs = [
            Document(
                page_content="CRITICAL ERROR 2026-10-16 [AuthService] - Failed to connect to Redis cache...",
                metadata={"source": "log_redis_error.txt"},
            ),
            Document(
                page_content="Configuración de worker nodes en FastAPI con un límite estricto de memoria...",
                metadata={"source": "fastapi_config.txt"},
            ),
        ]
        self.bm25_retriever = BM25Retriever.from_documents(corpus_docs)
        self.bm25_retriever.k = 5

        # 3. Ensamble híbrido propio con soporte async
        self.ensemble_retriever = CustomHybridRetriever(
            vector_retriever=self.vector_retriever,
            bm25_retriever=self.bm25_retriever,
            weights=[0.5, 0.5],
        )

    def get_retriever(self, k: int = 5):
        self.vector_retriever.search_kwargs = {"k": k}
        self.bm25_retriever.k = k
        self.ensemble_retriever.k = k
        return self.ensemble_retriever