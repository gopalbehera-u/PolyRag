from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from config import get_settings
from services.vectorstore import get_retriever
from utils.logger import get_logger

logger = get_logger(__name__)

_settings = get_settings()

_llm = ChatGoogleGenerativeAI(
    model=_settings.gemini_model,
    google_api_key=_settings.gemini_api_key,
    temperature=0.2,
)

_ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are an AI document assistant.
Answer the question using ONLY the context below. Each context block is
labeled with its source (and page, if available). When you use information
from a block, mention which source it came from.

If the answer is not in the context, say exactly:
"I couldn't find that information in the document."

Context:
{context}

Question:
{question}
"""
)

_SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    """You are an AI document assistant.
Summarize the following document in clear, well-organized bullet points.

Document:
{text}
"""
)


def _format_docs(docs: List[Document]) -> str:
    blocks = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        label = source if page is None else f"{source} (page {page})"
        blocks.append(f"[{label}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def ask_document(question: str, sources: Optional[List[str]] = None, k: int = 4) -> dict:
    """Returns {"answer": str, "citations": [{"source", "page"}]}"""
    retriever = get_retriever(sources=sources, k=k)
    docs = retriever.invoke(question)

    if not docs:
        return {"answer": "I couldn't find that information in the document.", "citations": []}

    context = _format_docs(docs)
    chain = _ANSWER_PROMPT | _llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    citations = [
        {"source": d.metadata.get("source"), "page": d.metadata.get("page")}
        for d in docs
    ]
    return {"answer": answer, "citations": citations}


def generate_summary(text: str) -> str:
    chain = _SUMMARY_PROMPT | _llm | StrOutputParser()
    return chain.invoke({"text": text})