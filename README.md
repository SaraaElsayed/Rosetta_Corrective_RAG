# 𓂀 ROSETTA — Corrective RAG

A Retrieval-Augmented Generation system with a self-correction layer. When retrieved documents aren't relevant enough, the pipeline rewrites the query and falls back to live web search — so answers stay grounded whether the knowledge base has the answer or not.

> Built around ancient Egypt as a domain, but the architecture is general-purpose.

🟢 **[Live Demo →](https://huggingface.co/spaces/Saraay/Rosetta)**

---

## How It Works

\`\`\`
Question
   │
   ▼
Vector Store (Chroma)  ──→  Evaluate relevance (LLM)
                                 │
                    ┌────────────┴────────────┐
                    │                         │
               Correct docs            No correct docs
                    │                         │
           Knowledge Refinement        Rewrite query
           (top-3 sentences by              │
            cosine similarity)        Web Search (Tavily)
                    │                         │
                    └────────────┬────────────┘
                                 │
                           Generate Answer
\`\`\`

1. **Retrieve** — pull top-k chunks from ChromaDB using Cohere embeddings
2. **Evaluate** — LLM labels each chunk as Correct / Ambiguous / Incorrect
3. **Refine or Search** — correct chunks are filtered to the most relevant sentences; if none pass, the question is rewritten and Tavily fills the gap
4. **Generate** — Groq (LLaMA) produces the final answer from the assembled context

---

## Project Structure

\`\`\`
corrective-rag/
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .env.example.app
├── src/
│   ├── config.py          # API keys, paths, model names
│   ├── ingest.py          # Load Wikipedia PDFs → ChromaDB
│   ├── vector_store.py    # Cohere embeddings + Chroma retrieval
│   ├── llm.py             # Evaluate, refine, search, generate
│   ├── pipeline.py        # Corrective RAG orchestration
│   ├── prompts.py         # Prompt templates
│   ├── main.py            # FastAPI endpoint (/rag)
├── chroma_db/             # Persisted vector store (git-ignored)
└── requirements.txt
\`\`\`

---

## Stack

| Layer | Tool |
|---|---|
| Embeddings | Cohere `embed-english-v3.0` |
| Vector Store | ChromaDB (local persistence) |
| LLM | Groq — LLaMA 3 |
| Web Search fallback | Tavily |
| API | FastAPI |
| Demo UI | Gradio (Hugging Face Spaces) |
| Chunking | LangChain SemanticChunker + sliding window fallback |

---

## Getting Started

### 1. Clone & configure

```bash
git clone https://github.com/SaraaElsayed/Rosetta_Corrective_RAG.git
cd corrective-rag
cp docker/.env.example.app src/.env
```

Fill in `src/.env`:

```env
COHERE_API_KEY=...
GROQ_API_KEY=...
TAVILY_API_KEY=...
CHROMA_DB_PATH=../chroma_db
COHERE_EMBED_MODEL=embed-english-v3.0
GROQ_MODEL=llama3-8b-8192
```

### 2. Ingest documents

```bash
cd src
pip install -r ../requirements.txt
python ingest.py
```

This fetches Wikipedia articles, chunks them semantically, and stores embeddings in `chroma_db/`.

### 3a. Run with Docker

```bash
cd docker
docker-compose up --build
```

API available at `http://localhost:8000`. Try it:

```bash
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{"question": "Who was Ramesses II?"}'
```

### 3b. Run locally (no Docker)

```bash
cd src
uvicorn main:app --reload
```

---

## API

**`POST /rag`**

```json
// Request
{ "question": "How were the pyramids built?" }

// Response
{
  "answer": "...",
  "used_docs": [
    {
      "content": ["sentence 1", "sentence 2"],
      "source": "vector_store"
    }
  ]
}
```

`source` is either `"vector_store"` or `"web_search"`. When it's web search, a `search_query` field shows the rewritten query.

---

## Notes

- **Chunking strategy** — documents are split using LangChain's `SemanticChunker`, which groups sentences by meaning rather than fixed size. Any chunk that still exceeds 3000 characters after semantic splitting is further broken down with a sliding window (1000-char chunks, 100-char overlap) to keep retrieval precise without losing context at boundaries.
