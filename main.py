import asyncio
import os
import sys

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from schemas import RespuestaRAG

load_dotenv()

PERSIST_DIRECTORY = "./vectorstore"


def obtener_retriever():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No se encontró OPENAI_API_KEY en las variables de entorno.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
    )
    # top_k entre 3 y 5 según consigna
    return vectorstore.as_retriever(search_kwargs={"k": 3})


async def get_rag_response(query: str) -> RespuestaRAG:
    """Función asíncrona que ejecuta el flujo RAG grounded."""
    api_key = os.getenv("OPENAI_API_KEY")
    retriever = obtener_retriever()

    # 1. Recuperación de fragmentos relevantes
    docs = await retriever.ainvoke(query)
    contexto_texto = "\n\n".join([d.page_content for d in docs])

    # 2. Prompt Grounded (Filtro de veracidad)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Sos un asistente técnico estricto. Responde a la pregunta del usuario utilizando "
                    "EXCLUSIVAMENTE el siguiente CONTEXTO recuperado. "
                    "Si la respuesta no se encuentra explícitamente en el contexto, debes responder "
                    "que no tienes acceso a esa información o que no lo sabes. No inventes ni asumas información."
                ),
            ),
            (
                "human",
                "CONTEXTO RECUPERADO:\n{contexto}\n\nPREGUNTA DEL USUARIO:\n{pregunta}",
            ),
        ]
    )

    # 3. Modelo con salida estructurada Pydantic y resiliencia
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=api_key)
    structured_llm = llm.with_structured_output(RespuestaRAG).with_retry(
        stop_after_attempt=3
    )

    # 4. Cadena LCEL
    chain = prompt | structured_llm

    # 5. Ejecución asíncrona
    resultado: RespuestaRAG = await chain.ainvoke(
        {"contexto": contexto_texto, "pregunta": query}
    )

    # Inyectar las fuentes encontradas al objeto resultante
    resultado.fuentes_utilizadas = [d.page_content for d in docs]
    return resultado


async def main():
    print("--------------------------------------------------")
    print("🧪 PRUEBA 1: Pregunta cuya respuesta ESTÁ en el documento")
    print("--------------------------------------------------")
    q1 = "¿Cuál es el límite de memoria asignado a los worker nodes de FastAPI?"
    print(f"Pregunta: {q1}")
    res1 = await get_rag_response(q1)
    print("\nRespuesta estructurada:")
    print(res1.model_dump_json(indent=2))

    print("\n--------------------------------------------------")
    print("🧪 PRUEBA 2: Pregunta trampa (NO está en el documento)")
    print("--------------------------------------------------")
    q2 = "¿Cuál es la clave de WiFi de la oficina central de Telecom?"
    print(f"Pregunta: {q2}")
    res2 = await get_rag_response(q2)
    print("\nRespuesta estructurada:")
    print(res2.model_dump_json(indent=2))


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())