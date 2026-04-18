# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union, Optional
from pipeline import corrective_rag

app = FastAPI(title="Corrective RAG")


class RAGRequest(BaseModel):
    question: str

class SourceDoc(BaseModel):
    content: Union[str, List[str]]
    source: str
    search_query: Optional[str] = None

class RAGResponse(BaseModel):
    answer: str
    used_docs: List[SourceDoc] = []


@app.post("/rag", response_model=RAGResponse)
def rag_endpoint(req: RAGRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")
    answer, used_docs = corrective_rag(req.question)
    return RAGResponse(answer=answer, used_docs=used_docs)