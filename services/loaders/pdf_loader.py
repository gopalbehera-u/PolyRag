import tempfile
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from config import get_settings
from services.loaders.image_loader import extract_text_from_image
from utils.logger import get_logger

logger = get_logger(__name__)


def extract_documents_from_pdf(pdf_path: str) -> List[Document]:
    """
    Returns a list of LangChain Document objects, one per page, with
    metadata = {"source": <filename>, "page": <page_number>}.
    Pages with no extractable native text fall back to OCR.
    """
    settings = get_settings()
    source_name = Path(pdf_path).name

    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()  # one Document per page, 0-indexed page in metadata

    documents: List[Document] = []
    pages_needing_ocr: List[int] = []

    for doc in raw_docs:
        page_num = doc.metadata.get("page", 0) + 1  # convert to 1-indexed
        text = doc.page_content.strip()

        if text:
            documents.append(
                Document(page_content=text, metadata={"source": source_name, "page": page_num})
            )
        else:
            pages_needing_ocr.append(page_num)

    if pages_needing_ocr:
        logger.info(
            "PDF %s: %d page(s) had no extractable text, running OCR fallback on %s",
            source_name, len(pages_needing_ocr), pages_needing_ocr,
        )
        ocr_docs = _ocr_fallback(pdf_path, pages_needing_ocr, source_name, settings.poppler_path)
        documents.extend(ocr_docs)

    documents.sort(key=lambda d: d.metadata["page"])
    return documents


def _ocr_fallback(pdf_path: str, page_numbers: List[int], source_name: str, poppler_path) -> List[Document]:
    from pdf2image import convert_from_path

    results: List[Document] = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for page_num in page_numbers:
            images = convert_from_path(
                pdf_path, first_page=page_num, last_page=page_num, poppler_path=poppler_path
            )
            if not images:
                continue
            image_path = str(Path(temp_dir) / f"page_{page_num}.png")
            images[0].save(image_path)
            text = extract_text_from_image(image_path)
            if text.strip():
                results.append(
                    Document(page_content=text, metadata={"source": source_name, "page": page_num})
                )
    return results