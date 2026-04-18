# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent

load_dotenv()

CHROMA_DB_PATH = str(BASE_DIR / "chroma_db")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama3-8b-8192")

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
COHERE_EMBED_MODEL = os.environ.get("COHERE_EMBED_MODEL", "embed-english-light-v3.0")

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

PDF_DIR = str(BASE_DIR / "data")

if not GROQ_API_KEY:
    raise RuntimeError("Set GROQ_API_KEY in env")
if not COHERE_API_KEY:
    raise RuntimeError("Set COHERE_API_KEY in env (free trial at cohere.com)")