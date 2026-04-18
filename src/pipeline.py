# pipeline.py
from typing import List, Tuple
from vector_store import retrieve_docs
from llm import evaluate_docs, rewrite_question, knowledge_refinement, web_search, generate_answer


def corrective_rag(question: str) -> Tuple[str, List[str]]:
    """
    Corrective RAG pipeline:
      1. Retrieve docs
      2. Evaluate relevance
      3. If any correct → refine; otherwise → web search fallback
      4. Generate answer
    """
    # 1. Retrieve
    retrieved = retrieve_docs(question, k=3)

    # 2. Evaluate
    evaluated = evaluate_docs(retrieved, question)
    correct = [doc for doc, label in evaluated if label == "Correct"]
    ambiguous = [doc for doc, label in evaluated if label == "Ambiguous"]

    # 3. Build context
    docs=[]
    if correct:
        final_docs = []
        for doc in correct:
            final_docs.extend(knowledge_refinement(doc, question))
        docs.append({
            "content": final_docs,
            "source": "vector_store"
        })   
    else:
        query = rewrite_question(question)
        final_docs = ambiguous + web_search(query)
        docs.append({
        "content": final_docs,
        "source": "web_search",
        "search_query": query
    })

    # 4. Generate
    answer = generate_answer(final_docs, question)
    return answer, docs