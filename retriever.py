import os

from dotenv import load_dotenv
from langchain.retrievers.ensemble import EnsembleRetriever  # type: ignore
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# Validación estricta de variables de entorno
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("❌ Error crítico: Falta la variable OPENAI_API_KEY en el entorno.")
if not os.getenv("PINECONE_API_KEY"):
    raise ValueError("❌ Error crítico: Falta la variable PINECONE_API_KEY en el entorno.")


class RAGSystem:
    """Sistema RAG Híbrido (Vectorial + BM25) encapsulado para Coderhouse."""

    def __init__(self, data_path: str = "data/manual_telecom.txt", k: int = 5):
        self.index_name = os.getenv("INDEX_NAME", "rag-tecnico-index")
        self.k = k

        # Inicializar embeddings de OpenAI
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # Conexión al vector store en Pinecone (Cloud)
        self.vectorstore = PineconeVectorStore(
            index_name=self.index_name, embedding=self.embeddings
        )
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": self.k}
        )

        # Cargar y splitear los documentos reales para el BM25
        if os.path.exists(data_path):
            loader = TextLoader(data_path, encoding="utf-8")
            raw_docs = loader.load()
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=50
            )
            corpus_docs = text_splitter.split_documents(raw_docs)
        else:
            corpus_docs = [
                Document(page_content="Documentación técnica de respaldo por defecto de Telecom.")
            ]

        # Recuperador Léxico BM25 basado en los chunks reales
        self.bm25_retriever = BM25Retriever.from_documents(corpus_docs)
        self.bm25_retriever.k = self.k

        # Ensemble Retriever real (50% Semántico / 50% Léxico)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=[0.5, 0.5]
        )

    def get_retriever(self, k: int | None = None):
        """Devuelve el retriever configurado para consultas."""
        if k:
            self.k = k
            self.vector_retriever.search_kwargs = {"k": k}
            self.bm25_retriever.k = k
        return self.ensemble_retriever