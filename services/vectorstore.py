from typing import Dict,List , Optional

from langchain_chroma import Chroma

from  langchain_core.documents import Document


from config import get_settings
from services.embeddings import get_embedding_model
from utils.logger import get_logger


logger=get_logger(__name__)

_settings=get_settings()

_vectorstores=Chroma(
    collection_name="documents",
    embedding_function=get_embedding_model(),
    persist_directory=str(_settings.chroma_persist_dir),
)


def store_document(documents : list[Document]) -> int:
     """Embed and store a list of LangChain Documents. Returns count stored."""
     if not documents:
          logger.warning("store_documents called with empty document list")
          return 0
     ids=[f"{doc.metadata.get('source', 'unknown')}::{doc.metadata.get('page', 0)}::{i}"
           for i, doc in enumerate(documents)]
     _vectorstores.add_documents(documents=documents,ids=ids)
     logger.info("Stored %d chunks (source=%s)", len(documents), documents[0].metadata.get("source"))
     return len(documents)



def remove_document(source:str)-> None:
        try:
            existing = _vectorstores.get(where={"source": source})
            ids = existing.get("ids") if existing else None
            if ids:
                _vectorstores.delete(ids=ids)
                logger.info("Removed %d chunks for source=%s", len(ids), source)
            else:
                logger.info("No chunks found for source=%s", source)
        except Exception:
            logger.exception("Failed to remove document source=%s", source)
            raise


def clear_all() -> None:
    try:
        existing = _vectorstores.get()
        ids = existing.get("ids") if existing else None
        if ids:
            _vectorstores.delete(ids=ids)
            logger.info("Cleared all %d stored chunks", len(ids))
    except Exception:
        logger.exception("Failed to clear all documents")
        raise


def list_sources() -> List[Dict]:
    existing = _vectorstores.get()
    counts: Dict[str, int] = {}
    for meta in existing.get("metadatas") or []:
        src = meta.get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1
    return [{"source": s, "chunks": c} for s, c in sorted(counts.items())]


def search(query: str, sources: Optional[List[str]] = None, k: int = 4) -> List[Document]:
    filter_dict = None
    if sources:
        filter_dict = {"source": sources[0]} if len(sources) == 1 else {"source": {"$in": sources}}

    return _vectorstores.similarity_search(query, k=k, filter=filter_dict)





def get_retriever(sources: Optional[List[str]] = None, k: int = 4):
    """Returns a LangChain retriever object for use in LCEL chains."""
    search_kwargs = {"k": k}
    if sources:
        search_kwargs["filter"] = {"source": sources[0]} if len(sources) == 1 else {"source": {"$in": sources}}
    return _vectorstores.as_retriever(search_kwargs=search_kwargs)