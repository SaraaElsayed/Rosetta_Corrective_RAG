# prompts.py
# All LLM prompt templates live here — easy to tune without touching logic.

def evaluate_doc_prompt(question: str, doc: str) -> str:
    return f"""### Instruction
Analyze the relationship between the provided Question and Document. Your goal is to determine if a user reading ONLY the Document could provide a comprehensive answer to the Question.

### Classification Criteria
- **Correct**: The document provides direct facts that answer the core of the question. Even if it doesn't cover the subject's entire life, it provides identifying historical actions and roles.
- **Ambiguous**: The document mentions the subject but lacks specific details to answer "who" they were, OR the information is suggestive but not explicit.
- **Incorrect**: The document is about a different subject, or does not contain information relevant to the question.

### Input Data
Question: {question}
Document: {doc}

### Response
Answer with exactly one word from: Correct, Incorrect, Ambiguous."""


def rewrite_question_prompt(question: str) -> str:
    return f"""Extract up to three short keywords/phrases (comma-separated) suitable as a web search query for:
Question: {question}
Query:"""


def generation_prompt(context: str, question: str) -> str:
    return f"""You are an expert Q&A assistant. 

### INSTRUCTION:
Answer the following Question based primarily on the provided Context. 
If the information is not present in the Context, state that you do not have enough information.

### QUESTION: 
{question}

### CONTEXT:
{context}

### FINAL ANSWER (Up to 3 sentences):"""