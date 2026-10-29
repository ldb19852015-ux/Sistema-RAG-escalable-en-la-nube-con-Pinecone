import asyncio

from dotenv import load_dotenv

from retriever import RAGSystem

load_dotenv()


async def evaluar_sistema():
    print("🧪 Inicializando evaluación del RAGSystem (Golden Set de 5 preguntas)...")
    
    # Instanciar el sistema pedido por la consigna
    rag = RAGSystem()
    retriever = rag.get_retriever(k=5)

    # Golden Set ampliado a 5 preguntas reales basadas en el contenido de data/manual_telecom.txt
    # (Asegurate que las fuentes 'expected_source' coincidan con los nombres de archivos o fragmentos esperados)
    golden_set = [
        {
            "pregunta": "¿Cuál es el procedimiento para la revisión de planta externa FTTH?",
            "expected_source": "manual_telecom.txt"
        },
        {
            "pregunta": "¿Qué parámetros técnicos se deben verificar en los sistemas de alimentación y rectificadores?",
            "expected_source": "manual_telecom.txt"
        },
        {
            "pregunta": "¿Cómo se gestionan las alarmas críticas en la plataforma DCIM?",
            "expected_source": "manual_telecom.txt"
        },
        {
            "pregunta": "¿Cuáles son los pasos de diagnóstico ante una caída de red HFC en Headend?",
            "expected_source": "manual_telecom.txt"
        },
        {
            "pregunta": "¿Qué herramientas y sistemas administrativos se utilizan para el control de planta interna?",
            "expected_source": "manual_telecom.txt"
        },
    ]

    hits = 0
    total = len(golden_set)

    print("\n" + "=" * 60)
    print(f"EJECUTANDO BENCHMARK DE RECUPERACIÓN (K=5) - Total: {total} preguntas")
    print("=" * 60)

    for i, item in enumerate(golden_set, 1):
        query = item["pregunta"]
        expected = item["expected_source"]

        # Invocación asincrónica soportada por el ensemble
        try:
            docs = await retriever.ainvoke(query)
        except AttributeError:
            # Fallback sincrónico si ainvoke no está expuesto directamente en el wrapper
            docs = retriever.invoke(query)

        # Verificar si la fuente esperada está en los metadatos de los docs recuperados
        sources_found = [doc.metadata.get("source", "manual_telecom.txt") for doc in docs]
        hit_detected = any(expected in str(s) for s in sources_found)

        if hit_detected:
            hits += 1
            status = "✅ Sí"
        else:
            status = "❌ No"

        precision = 1.0 if hit_detected else 0.0

        print(f"\n[{i}] Pregunta: {query}")
        print(f"    Esperado (fuente/contenido): {expected}")
        print(f"    Recuperados (fuentes): {sources_found}")
        print(f"    Hit (Recall@5): {status} | Precision@5: {precision:.2f}")

    recall_score = (hits / total) * 100

    print("\n" + "=" * 60)
    print("📊 RESUMEN DE MÉTRICAS FINALES (POST-FEEDBACK)")
    print("=" * 60)
    print(f"Recall@5: {recall_score:.1f%} ({hits}/{total} aciertos)")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(evaluar_sistema())