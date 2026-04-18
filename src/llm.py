# llm.py
from typing import List
import numpy as np
import spacy
from groq import Groq
from scipy.spatial.distance import cosine
from langchain_community.tools.tavily_search import TavilySearchResults
import logging
from config import GROQ_API_KEY, GROQ_MODEL
from vector_store import CohereEmbeddings
import prompts as prompts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

_client = Groq(api_key=GROQ_API_KEY)


def _chat(prompt: str, max_tokens: int = 256) -> str:
    resp = _client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


def evaluate_docs(docs: List[str], question: str) -> List[tuple]:
    """Returns [(doc_text, label), ...] where label is Correct / Incorrect / Ambiguous."""
    results = []
    for i, doc in enumerate(docs):
        raw = _chat(prompts.evaluate_doc_prompt(question, doc), max_tokens=50)
        logger.info(f"Doc {i} Evaluation Raw Output: '{raw}'")

        raw_lower = raw.lower()
        logger.info(f"Doc {i} Evaluation lowerd Raw Output: '{raw_lower}'")
        
        if "incorrect" in raw_lower:
            label = "Incorrect"
        elif "correct" in raw_lower:
            label = "Correct"
        else:
            label = "Ambiguous"
        #label = raw.strip().splitlines()[0].strip().split()[-1].capitalize()
        if label != "Correct":
            label = "Ambiguous"

        logger.info(f"Final parsed label for Doc {i}: {label}")    
        logger.info(f"Doc {i}: {doc}") 
        results.append((doc, label))
    return results


def rewrite_question(question: str) -> str:
    """Turn a question into short web-search keywords."""
    return _chat(prompts.rewrite_question_prompt(question), max_tokens=60)


def knowledge_refinement(doc_text: str, question: str) -> List[str]:
    """
    Split doc into sentences, return top-3 by cosine similarity to the question.
    """
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    sentences = [s.text for s in nlp(doc_text).sents]
    if not sentences:
        return []

    embedder = CohereEmbeddings()
    q_vec = np.array(embedder.embed_query(question))
    s_vecs = [np.array(vec) for vec in embedder.embed_documents(sentences)]
    ranked = sorted(zip(sentences, s_vecs), key=lambda x: 1 - cosine(q_vec, x[1]), reverse=True)
    return [s for s, _ in ranked[:3]]


def web_search(query: str, k: int = 3) -> List[str]:
    try:
        results = TavilySearchResults(k=k).invoke({"query": query})
        return [r.get("content", "") for r in results]
    except Exception:
        return []


def generate_answer(documents: List[str], question: str) -> str:
    context = "\n\n".join(documents) if documents else ""
    return _chat(prompts.generation_prompt(context, question), max_tokens=256)