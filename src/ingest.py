# ingest_wikipedia.py
import wikipedia
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from vector_store import _embedding_model, _safe_chunks
from config import CHROMA_DB_PATH

TOPICS = [
    # Pharaohs
    "Ramesses II", "Tutankhamun", "Cleopatra", "Akhenaten",
    "Thutmose III", "Amenhotep II", "Hatshepsut", "Seti I",
    "Khufu", "Amenhotep III", "Thutmose IV",
    # Civilizations & Concepts
    "Ancient Egypt", "Egyptian pyramids", "Sphinx of Giza",
    "Egyptian mythology", "Nile River", "Valley of the Kings",
    "Rosetta Stone", "Hieroglyphics",
    "Mummification in ancient Egypt",
]

def fetch_topic(topic: str) -> str:
    try:
        # summary only, not full page — ~500 chars instead of 50,000
        return wikipedia.summary(topic, sentences=15, auto_suggest=False)
    except wikipedia.DisambiguationError as e:
        return wikipedia.summary(e.options[0], sentences=15, auto_suggest=False)
    except Exception as e:
        print(f"✗ {topic}: {e}")
        return ""

def ingest():
    embeddings = _embedding_model()
    all_documents = []

    for topic in TOPICS:
        raw_text = fetch_topic(topic)
        if not raw_text:
            continue
        chunks = _safe_chunks(raw_text, embeddings)
        for chunk in chunks:
            all_documents.append(Document(
                page_content=chunk,
                metadata={"source": topic}  # track which topic each chunk came from
            ))
        print(f"   → {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_documents)}")
    print("Storing in Chroma...")

    store = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )
    try:
        store.persist()
    except AttributeError:
        pass

    print("Done.")

if __name__ == "__main__":
    ingest()