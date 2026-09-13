import os
import re
from pathlib import Path

import streamlit as st
from langchain_core.documents import Document

from services.loaders.pdf_loader import extract_documents_from_pdf
from services.loaders.image_loader import extract_text_from_image
from services.cleaner import clean_text
from services.chunker import chunk_text
from services.llm import generate_summary, ask_document
from services.vectorstore import store_document, remove_document, clear_all, list_sources

st.set_page_config(page_title="PolyRAG", page_icon="📄", layout="wide")

# --------------------------------------------------
# Inline info extraction (no separate service file)
# --------------------------------------------------
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?<!\d)(\+?\d{1,3}[\s.-]?)?(\(?\d{2,4}\)?[\s.-]?){2,4}\d{2,4}(?!\d)")
_DATE_RE = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b")


def _valid_phone(candidate: str) -> bool:
    digits = re.sub(r"\D", "", candidate)
    return 8 <= len(digits) <= 15


def extract_information(text: str) -> dict:
    if not text:
        return {"Emails": [], "Phone Numbers": [], "Dates": []}
    emails = sorted(set(_EMAIL_RE.findall(text)))
    phone_candidates = [m.group(0).strip() for m in _PHONE_RE.finditer(text)]
    phones = sorted({p for p in phone_candidates if _valid_phone(p)})
    dates = sorted(set(_DATE_RE.findall(text)))
    return {"Emails": emails, "Phone Numbers": phones, "Dates": dates}


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.title("📄 PolyRAG")
    st.markdown("---")
    st.subheader("Supported Files")
    st.write("✅ PDF  ✅ JPG  ✅ JPEG  ✅ PNG")
    st.markdown("---")
    st.info("Upload one or more documents to begin.")
    st.markdown("---")
    st.subheader("About")
    st.write("AI-powered Document Intelligence System\n\n• OCR\n• AI Summary\n• Multi-Document Chat (RAG)\n• Information Extraction")

# --------------------------------------------------
# Directories
# --------------------------------------------------
UPLOAD_DIR = "data/uploads"
EXTRACT_DIR = "data/extracted"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)

# --------------------------------------------------
# Session State
# --------------------------------------------------
if "summaries" not in st.session_state:
    st.session_state.summaries = {}
if "extracted_texts" not in st.session_state:
    st.session_state.extracted_texts = {}
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()
if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------------
# Main Page
# --------------------------------------------------
st.title("📄 PolyRAG")
st.subheader("Intelligent Document Assistant")
st.write("Upload one or more PDFs or images to extract text, summarize, and chat across all of your documents at once.")
st.divider()

uploaded_files = st.file_uploader(
    "Upload PDF or Image files",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

col_a, col_b = st.columns([1, 5])
with col_a:
    if st.button("🗑️ Clear all documents"):
        clear_all()
        st.session_state.summaries = {}
        st.session_state.extracted_texts = {}
        st.session_state.indexed_files = set()
        st.session_state.messages = []
        st.rerun()

# --------------------------------------------------
# Process Documents
# --------------------------------------------------
if uploaded_files:
    progress = st.progress(0)
    total = len(uploaded_files)

    for idx, uploaded_file in enumerate(uploaded_files):
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        if uploaded_file.name not in st.session_state.indexed_files:
            extension = uploaded_file.name.split(".")[-1].lower()

            if extension == "pdf":
                page_docs = extract_documents_from_pdf(file_path)
                full_text = "\n\n".join(d.page_content for d in page_docs)
            else:
                raw_text = extract_text_from_image(file_path)
                full_text = raw_text
                page_docs = [Document(page_content=raw_text, metadata={"source": uploaded_file.name})]

            full_text = clean_text(full_text)
            st.session_state.extracted_texts[uploaded_file.name] = full_text

            with st.spinner(f"Indexing {uploaded_file.name}..."):
                chunk_docs = []
                for page_doc in page_docs:
                    cleaned = clean_text(page_doc.page_content)
                    for chunk in chunk_text(cleaned):
                        meta = {"source": uploaded_file.name}
                        if page_doc.metadata.get("page") is not None:
                            meta["page"] = page_doc.metadata["page"]
                        chunk_docs.append(Document(page_content=chunk.text, metadata=meta))
                store_document(chunk_docs)

            st.session_state.indexed_files.add(uploaded_file.name)

            text_path = os.path.join(EXTRACT_DIR, uploaded_file.name + ".txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(full_text)

        progress.progress(int(((idx + 1) / total) * 100))

    st.success(f"✅ {total} file(s) uploaded and indexed successfully!")

    all_filenames = list(st.session_state.extracted_texts.keys())

    tab1, tab2, tab3, tab4 = st.tabs(["📄 Extracted Text", "📝 AI Summary", "💬 Chat", "📌 Information"])

    with tab1:
        selected_doc = st.selectbox("Select a document", all_filenames, key="extract_select")
        doc_text = st.session_state.extracted_texts[selected_doc]
        st.text_area("Extracted Text", doc_text, height=450)
        st.download_button("⬇ Download Extracted Text", doc_text, file_name=f"{selected_doc}.txt", mime="text/plain")

    with tab2:
        summary_doc = st.selectbox("Select a document", all_filenames, key="summary_select")
        if st.button("📝 Generate AI Summary"):
            with st.spinner("Generating Summary..."):
                st.session_state.summaries[summary_doc] = generate_summary(
                    st.session_state.extracted_texts[summary_doc]
                )
        if st.session_state.summaries.get(summary_doc):
            st.markdown(st.session_state.summaries[summary_doc])
            st.download_button(
                "⬇ Download Summary", st.session_state.summaries[summary_doc],
                file_name=f"{summary_doc}_summary.txt", mime="text/plain",
            )

    with tab3:
        search_scope = st.multiselect("Search within (leave empty to search all documents)", all_filenames, default=[])

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.text_input("Ask anything about your document(s)", key="question_input")

        if st.button("Ask", key="ask_button") and question:
            with st.spinner("Searching..."):
                result = ask_document(question, sources=search_scope if search_scope else None)

            st.session_state.messages.append({"role": "user", "content": question})

            answer_with_sources = result["answer"]
            if result["citations"]:
                cite_lines = ", ".join(
                    f"{c['source']}" + (f" (p.{c['page']})" if c.get("page") else "")
                    for c in result["citations"]
                )
                answer_with_sources += f"\n\n*Sources: {cite_lines}*"

            st.session_state.messages.append({"role": "assistant", "content": answer_with_sources})
            st.rerun()

    with tab4:
        info_doc = st.selectbox("Select a document", all_filenames, key="info_select")
        info = extract_information(st.session_state.extracted_texts[info_doc])
        st.json(info)

    st.divider()
    st.subheader("📊 Document Statistics")
    combined_text = "\n".join(st.session_state.extracted_texts.values())
    s1, s2, s3 = st.columns(3)
    s1.metric("Documents", len(all_filenames))