# 📄 PolyRAG — Multi-Format Document Intelligence Assistant

PolyRAG is a Retrieval-Augmented Generation (RAG) system that lets you upload PDFs, DOCX files, plain text, and images (including scanned documents), then ask natural-language questions and get **cited, grounded answers** — instead of relying on an LLM's memory alone.

Built end-to-end with **LangChain**, **ChromaDB**, and **Google Gemini**, wrapped in a multi-tab **Streamlit** interface.

---

## ✨ Features

- **Multi-format ingestion** — PDF, DOCX, TXT/Markdown, and images (PNG/JPG)
- **OCR fallback** — scanned/image-only PDF pages are automatically detected and run through EasyOCR + OpenCV preprocessing
- **Multi-document support** — upload several files at once; search across all of them or scope to specific sources
- **Source & page-level citations** — every answer tells you exactly which document (and page, for PDFs) it came from
- **AI summarization** — generate a bullet-point summary of any indexed document on demand
- **Information extraction** — pulls emails, phone numbers, and dates out of any document via regex
- **Persistent vector storage** — documents stay indexed across app restarts (ChromaDB, disk-backed)
- **Chat-style Q&A** — multi-turn conversation interface, scoped search per question

---

## 🧠 How It Works

```
Upload File
   │
   ▼
Loader (format-specific)          PDF → LangChain PyPDFLoader (+ per-page OCR fallback)
   │                              DOCX → LangChain Docx2txtLoader
   │                              TXT/MD → LangChain TextLoader
   │                              Image → EasyOCR + OpenCV preprocessing
   ▼
Clean (normalize whitespace, preserve paragraph structure)
   ▼
Chunk (LangChain RecursiveCharacterTextSplitter, paragraph → sentence → word boundaries, with overlap)
   ▼
Embed (HuggingFace sentence-transformers, all-MiniLM-L6-v2)
   ▼
Store (ChromaDB, persistent, tagged with source + page metadata)
   ▼
──────────────────────────────────────
   ▼
Ask a Question
   ▼
Retrieve top-k relevant chunks (optionally scoped to specific sources)
   ▼
LCEL chain: prompt | Gemini (ChatGoogleGenerativeAI) | output parser
   ▼
Cited Answer
```

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Orchestration | LangChain (LCEL) | Composable, swappable pipeline components |
| Vector store | ChromaDB (via `langchain-chroma`), persistent | Embedded, no separate server, metadata filtering |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) via `langchain-huggingface` | Free, local, no per-call API cost |
| LLM | Google Gemini via `langchain-google-genai` | Fast, cheap, good multimodal support |
| PDF parsing | LangChain `PyPDFLoader` (PyMuPDF/pypdf backend) | Fast native text extraction |
| OCR | EasyOCR + OpenCV preprocessing | Offline, no cloud dependency, decent accuracy |
| DOCX parsing | LangChain `Docx2txtLoader` | Standard, reliable |
| UI | Streamlit | Fast to build, great for demos |
| Config | `pydantic-settings` + `.env` | Type-safe, single source of truth |

---

## 📁 Project Structure

```
PolyRAG/
├── app.py                     # Streamlit UI — upload, chat, summary, extraction, stats
├── config.py                  # Centralized settings (reads from .env)
├── services/
│   ├── cleaner.py              # Text normalization
│   ├── chunker.py               # LangChain RecursiveCharacterTextSplitter wrapper
│   ├── embeddings.py            # LangChain HuggingFaceEmbeddings wrapper
│   ├── ocr.py                   # EasyOCR wrapper
│   ├── vectorstore.py           # LangChain Chroma wrapper (store/search/delete)
│   ├── ingest.py                 # Orchestrates: load → clean → chunk → embed → store
│   ├── llm.py                    # Gemini + LCEL chains (ask_document, generate_summary)
│   └── loaders/
│       ├── pdf_loader.py          # PyPDFLoader + per-page OCR fallback
│       ├── docx_loader.py         # Docx2txtLoader
│       ├── text_loader.py         # TextLoader
│       └── image_loader.py        # OpenCV preprocessing + EasyOCR
├── utils/
│   └── logger.py                # Shared structured logging setup
├── data/
│   ├── uploads/                  # Raw uploaded files
│   └── chroma/                    # Persistent vector DB storage
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone and set up a virtual environment

```bash
git clone <your-repo-url>
cd PolyRAG
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# macOS/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Poppler (required for scanned-PDF OCR fallback)

Poppler is a system dependency (not a Python package) needed by `pdf2image`.

- **Windows:** download from [poppler-windows releases](https://github.com/oschwartz10612/poppler-windows/releases/latest) → download the `Release-XX.XX.X-X.zip` asset (not the source code zip) → extract → note the path to `Library/bin` inside it
- **macOS:** `brew install poppler`
- **Linux:** `sudo apt-get install poppler-utils`

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and set:

```env
GEMINI_API_KEY=your_real_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash-lite

# Windows only, if poppler isn't on PATH:
POPPLER_PATH=C:\poppler\poppler-26.02.0\Library\bin
```

Get a free Gemini API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

### 5. Run the app

```bash
streamlit run app.py
```

Open the local URL it prints (typically `http://localhost:8501`).

---

## 💡 Usage

1. **Upload** one or more PDFs, DOCX, TXT, or image files in the main upload area
2. Wait for indexing to complete (progress bar shows per-file status)
3. Use the tabs:
   - **📄 Extracted Text** — view/download raw extracted text per document
   - **📝 AI Summary** — generate a bullet-point summary for any document
   - **💬 Chat** — ask questions across all documents, or scope to specific ones; answers include source citations
   - **📌 Information** — auto-extracted emails, phone numbers, and dates
4. **🗑️ Clear all documents** — wipes the vector index and resets the session

---

## ⚠️ Known Limitations

- **OCR is slow on CPU** — large scanned PDFs (many image-only pages) can take minutes; EasyOCR runs faster with a CUDA-capable GPU (`gpu=True` in `services/ocr.py`)
- **Streamlit runs synchronously** — the UI blocks during ingestion; there's no background job queue or fine-grained progress within a single large file
- **No authentication** — intended for local/personal use as-is; add an auth layer before any public deployment
- **Chunking/embedding runs on CPU by default** — fine for typical documents, but large batches will be slower than a GPU-backed setup

---

## 🗺️ Possible Next Steps

- Background job queue (e.g., Celery/Redis) for large-file ingestion with real progress tracking
- Swap ChromaDB for a managed vector store (Pinecone/Weaviate) for multi-user, larger-scale deployments
- Add a FastAPI layer alongside Streamlit for programmatic/API access
- Containerize with Docker for reproducible deployment
- GPU-backed OCR for large scanned-document workloads

---

## 👤 Author

**Gopal Behera**
AIML Engineer

- GitHub: [github.com/your-username](https://github.com/gopalbehera-u)
- LinkedIn: [linkedin.com/in/your-profile](https://www.linkedin.com/in/gopal-behera-8b3676424/)