import time
from typing import List
import cohere
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
import os
from config import CHROMA_DB_PATH, COHERE_API_KEY, COHERE_EMBED_MODEL

_EMBED_INPUT_TYPE_DOC = "search_document"
_EMBED_INPUT_TYPE_QUERY = "search_query"

_MAX_CHUNK_CHARS = 3000
_FALLBACK_CHUNK_SIZE = 1000
_FALLBACK_CHUNK_OVERLAP = 100

class CohereEmbeddings(Embeddings):
    def __init__(self, api_key: str = COHERE_API_KEY, model: str = COHERE_EMBED_MODEL):
        self._client = cohere.Client(api_key)
        self._model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds documents in batches to avoid Cohere Trial Rate Limits."""
        all_embeddings = []
        batch_size = 10  # Smaller batches help stay under Token Per Minute limits
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                resp = self._client.embed(
                    texts=batch, 
                    model=self._model, 
                    input_type=_EMBED_INPUT_TYPE_DOC
                )
                all_embeddings.extend(resp.embeddings)
                # Small pause to avoid hitting the 100k TPM (Tokens Per Minute) limit
                time.sleep(2) 
            except cohere.TooManyRequestsError:
                print("Rate limit hit, sleeping for 10 seconds...")
                time.sleep(10)
                # Simple retry logic for the current batch
                resp = self._client.embed(
                    texts=batch, 
                    model=self._model, 
                    input_type=_EMBED_INPUT_TYPE_DOC
                )
                all_embeddings.extend(resp.embeddings)
                
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        resp = self._client.embed(texts=[text], model=self._model, input_type=_EMBED_INPUT_TYPE_QUERY)
        return resp.embeddings[0]

def _embedding_model() -> CohereEmbeddings:
    return CohereEmbeddings()

def _split_large_chunk(text: str) -> List[str]:
    parts = []
    start = 0
    while start < len(text):
        end = min(start + _FALLBACK_CHUNK_SIZE, len(text))
        parts.append(text[start:end])
        start += _FALLBACK_CHUNK_SIZE - _FALLBACK_CHUNK_OVERLAP
    return parts

def _safe_chunks(raw_text: str, embeddings: CohereEmbeddings) -> List[str]:
    try:
        # Note: SemanticChunker ALSO calls the embedding model internally.
        # If your raw_text is huge, this line might still trigger a 429.
        semantic_chunks = SemanticChunker(embeddings).split_text(raw_text)
    except Exception as e:
        print(f"SemanticChunker failed: {e}. Falling back to sliding window.")
        return _split_large_chunk(raw_text)

    result = []
    for chunk in semantic_chunks:
        if len(chunk) <= _MAX_CHUNK_CHARS:
            result.append(chunk)
        else:
            result.extend(_split_large_chunk(chunk))
    return result

def store_docs(pdf_paths: List[str]) -> int:
    raw_text = ""
    for path in pdf_paths:
        try:
            pages = PyMuPDFLoader(path).load()
            raw_text += "\n".join(p.page_content for p in pages) + "\n"
        except Exception as e:
            print(f"Error loading {path}: {e}")

    embeddings = _embedding_model()
    chunks = _safe_chunks(raw_text, embeddings)
    documents = [Document(page_content=c) for c in chunks]

    # Chroma.from_documents calls embed_documents internally, 
    # which now uses our batched logic.
    store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )
    # Note: Chroma v0.4.x+ persists automatically; 
    # newer versions may throw an error on .persist()
    try:
        store.persist()
    except AttributeError:
        pass 
        
    return len(documents)

def retrieve_docs(question: str, k: int = 3) -> List[str]:
    store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=_embedding_model())
    results = store.similarity_search(question, k=k)
    return [doc.page_content for doc in results]

