from dataclasses import dataclass
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_settings

@dataclass
class Chunk:
    text: str
    index: int


def chunk_text(text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[Chunk]:
    """Split `text` into overlapping chunks, preferring natural boundaries."""
    if not text or not text.strip():
        return []

    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 4

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    pieces = splitter.split_text(text.strip())

    return [Chunk(text=p, index=i) for i, p in enumerate(pieces) if p.strip()]