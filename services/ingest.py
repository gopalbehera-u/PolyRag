from pathlib import Path

from typing import List

from langchain_core.documents import Document

from services.chunker import chunk_text
from services.loaders.docx_loader import extract_documents_from_docx
from services.loaders.image_loader import extract_text_from_image
from services.loaders.pdf_loader import extract_documents_from_pdf
from services.loaders.text_loader import extract_documents_from_txt
from services.vectorstore import store_document
from utils.logger import get_logger


logger = get_logger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
SUPPORTED_EXTS = {".pdf", ".docx", ".txt", ".md"} | IMAGE_EXTS



def ingest_file(file_path: str) -> int:
    """
    Process a single file end-to-end: load -> chunk -> embed -> store.
    Returns the number of chunks stored.
    """
    ext = Path(file_path).suffix.lower()
    source_name = Path(file_path).name

    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTS)}")

    logger.info("Ingesting file=%s ext=%s", source_name, ext)

    # Step 1: Load -> list of (page_content, page_number_or_None) pairs
    if ext == ".pdf":
        page_docs = extract_documents_from_pdf(file_path)
        segments = [(d.page_content, d.metadata.get("page")) for d in page_docs]
    elif ext == ".docx":
        page_docs = extract_documents_from_docx(file_path)
        segments = [(d.page_content, None) for d in page_docs]
    elif ext in (".txt", ".md"):
        page_docs = extract_documents_from_txt(file_path)
        segments = [(d.page_content, None) for d in page_docs]
    elif ext in IMAGE_EXTS:
        text = extract_text_from_image(file_path)
        segments = [(text, None)] if text.strip() else []
    else:
        segments = []

    if not segments:
        logger.warning("No text extracted from %s", source_name)
        return 0

    # Step 2: Chunk each segment, preserving page metadata
    all_chunks: List[Document] = []
    for text, page in segments:
        chunks = chunk_text(text)
        for chunk in chunks:
            metadata = {"source": source_name}
            if page is not None:
                metadata["page"] = page
            all_chunks.append(Document(page_content=chunk.text, metadata=metadata))

    # Step 3: Store in vector DB
    stored = store_document(all_chunks)
    logger.info("Finished ingesting %s -> %d chunks stored", source_name, stored)
    return stored