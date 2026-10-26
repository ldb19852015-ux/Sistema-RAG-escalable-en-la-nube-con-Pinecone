import asyncio

from dotenv import load_dotenv

from retriever import HybridRAGRetriever

load_dotenv()

# Golden Set: preguntas de prueba y el archivo fuente esperado
GOLDEN_SET = [
    {
        "pregunta": "¿Cuál es el error crítico reportado en el servicio de autenticación con Redis?",
        "fuente_esperada": "log_redis_error.txt"
    },
    {
        "pregunta": "¿Cuál es el límite de memoria asignado a los worker nodes de FastAPI?",
        "fuente_esperada": "fastapi_config.txt"
    }
]


async def evaluar_sistema():
    print("🧪 Inicializando evaluación del Recuperador Híbrido...")
    hybrid_system = HybridRAGRetriever()
    retriever = hybrid_system.get_retriever(k=5)

    total_queries = len(GOLDEN_SET)
    recall_hits = 0
    total_precision = 0.0

    print("\n" + "="*60)
    print("EJECUTANDO BENCHMARK DE RECUPERACIÓN (K=5)")
    print("="*60)

    for i, item in enumerate(GOLDEN_SET, 1):
        query = item["pregunta"]
        expected_source = item["fuente_esperada"]

        # Recuperación asíncrona
        docs = await retriever.ainvoke(query)
        retrieved_sources = [doc.metadata.get("source") for doc in docs]

        # Recall@5: ¿Aparece la fuente esperada en los 5 resultados?
        hit = expected_source in retrieved_sources
        if hit:
            recall_hits += 1

        # Precision@5: Porcentaje de documentos recuperados que coinciden con el esperado
        relevant_matches = sum(1 for src in retrieved_sources if src == expected_source)
        precision = relevant_matches / len(retrieved_sources) if retrieved_sources else 0.0
        total_precision += precision

        print(f"[{i}] Pregunta: {query}")
        print(f"    Esperado: {expected_source}")
        print(f"    Recuperados: {retrieved_sources}")
        print(f"    Hit (Recall@5): {'✅ Sí' if hit else '❌ No'} | Precision@5: {precision:.2f}\n")

    recall_at_5 = (recall_hits / total_queries) * 100
    mean_precision_at_5 = (total_precision / total_queries) * 100

    print("="*60)
    print("📊 RESUMEN DE MÉTRICAS FINALES")
    print("="*60)
    print(f"Recall@5: {recall_at_5:.1f}% ({recall_hits}/{total_queries} aciertos)")
    print(f"Precision@5 Promedio: {mean_precision_at_5:.1f}%")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(evaluar_sistema())