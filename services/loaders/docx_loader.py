from pathlib import Path

from typing import List

from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document

def extract_documents_from_docx(docx_path : str) -> List[Document]:
    source_name=Path(docx_path).name
    loader=Docx2txtLoader(docx_path)
    raw_docs=loader.load()

    documents=[]

    for doc in raw_docs:
        text=doc.page_content.strip()
        if text:
            documents.append(
                Document(page_content=text,metadata={"source":source_name})
                                        )
    return documents