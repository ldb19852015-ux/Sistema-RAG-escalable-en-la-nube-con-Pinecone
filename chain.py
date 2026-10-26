import asyncio
import logging
import os
import sys

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from schemas import ExtraccionTecnica

# Configuración de logs para observar validaciones y reintentos
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

load_dotenv()


def obtener_cadena_lcel():
    """Construye y retorna la cadena LCEL estructurada con resiliencia."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No se encontró OPENAI_API_KEY en las variables de entorno.")

    # 1. Configuración del LLM
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        api_key=api_key,
    )

    # 2. Configuración de Salida Estructurada basada en Pydantic
    structured_llm = model.with_structured_output(ExtraccionTecnica)

    # 3. Lógica de Resiliencia: Reintentos automáticos (.with_retry)
    structured_llm_with_retry = structured_llm.with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=True,
    )

    # 4. ChatPromptTemplate modular
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Sos un experto en auditoría de sistemas y análisis de logs técnicos. "
                    "Tu tarea es analizar el texto proporcionado y extraer la información "
                    "técnica requerida siguiendo estrictamente la estructura solicitada."
                ),
            ),
            ("human", "Texto a analizar:\n\"\"\"{texto_entrada}\"\"\""),
        ]
    )

    # 5. Cadena LCEL: prompt | model.with_structured_output().with_retry()
    chain = prompt | structured_llm_with_retry
    return chain


async def process_text(text: str) -> ExtraccionTecnica:
    """Función asíncrona que procesa un texto mediante la cadena LCEL."""
    logger.info("Iniciando procesamiento de texto...")
    chain = obtener_cadena_lcel()

    try:
        # Ejecución asíncrona con .ainvoke()
        resultado: ExtraccionTecnica = await chain.ainvoke({"texto_entrada": text})
        logger.info("Procesamiento y validación exitosos.")
        return resultado
    except Exception as e:
        logger.error(f"Error durante el procesamiento o validación: {e}")
        raise


async def main():
    # Mini-script de prueba con un log de error técnico
    texto_prueba = (
        "CRITICAL ERROR 2026-10-16 08:30:12 [AuthService] - Failed to connect to Redis cache "
        "at 10.0.0.12:6379 (Timeout). Falling back to PostgreSQL main instance. "
        "High memory consumption detected in FastAPI worker nodes due to unhandled concurrent requests."
    )

    print("\n--- INICIANDO PRUEBA DEL PIPELINE ---")
    resultado = await process_text(texto_prueba)

    print("\n--- RESULTADO VALIDADO (Objeto Pydantic / JSON) ---")
    print(resultado.model_dump_json(indent=2))


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())