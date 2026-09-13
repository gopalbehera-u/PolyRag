from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader

from langchain_core.documents import Document

def extract_documents_from_txt(txt_path:str) -> List[Document]:
    source_name=Path(txt_path).name

    loader=TextLoader(txt_path,encoding='utf-8',autodetect_encoding=True)

    raw_docs=loader.load()

    documents=[]

    for doc in raw_docs:
        text=doc.page_content.strip()
        if text:
            documents.append(
                Document(page_content=text,metadata={'source':source_name})
            )
    return documents



