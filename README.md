# Pre-entrega 4: Sistema RAG Escalable en la Nube con Pinecone y Búsqueda Híbrida

Evolución del sistema RAG local de la pre-entrega 3, migrado a una arquitectura cloud-native con **Pinecone Serverless**, incorporando un **Recuperador Híbrido** (Vectorial + BM25) y un módulo de **Evaluación de Rendimiento**.

## 🚀 Componentes del Sistema
1. **Pipeline de Ingesta en la Nube (`ingest.py`)**: Valida o crea un índice Serverless en Pinecone (`dimension=1536`), fragmenta los documentos y almacena el contenido en los metadatos.
2. **Recuperador Híbrido (`retriever.py`)**: Combina la similitud semántica densa de Pinecone con la coincidencia exacta de palabras clave vía BM25 mediante `EnsembleRetriever`.
3. **Módulo de Evaluación (`evaluate.py`)**: Corre un Golden Set automatizado imprimiendo métricas cuantitativas de **Recall@5** y **Precision@5**.

## 🛠️ Instrucciones de Configuración y Ejecución

1. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt