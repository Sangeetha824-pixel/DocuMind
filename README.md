# DocuMind - PDF RAG Chatbot

DocuMind is a professional Flask-based PDF question-answering app. Upload a PDF,
index it locally with TF-IDF retrieval, and ask grounded questions through a clean
chat workspace.

## Features

- PDF upload with secure filename handling and a 25 MB size limit
- Text extraction with `pypdf`
- Chunking with overlap for better retrieval context
- Local TF-IDF indexing with `scikit-learn`
- Cosine-similarity source retrieval
- Optional Claude answer generation through `ANTHROPIC_API_KEY`
- Source previews and similarity scores in the chat UI
- Persisted document indexes in `vectorstore/`
- Responsive professional interface for desktop and mobile

## How It Works

1. Upload a PDF.
2. The app extracts text and splits it into overlapping chunks.
3. Chunks are vectorized locally with TF-IDF.
4. Your question is matched against the most relevant chunks.
5. If Claude is configured, the answer is generated from the retrieved context.
6. If no API key is configured, the app still returns the most relevant excerpts.

## Project Structure

```text
pdf_rag_chatbot/
|-- app.py              # Flask routes for upload, documents, and chat
|-- rag.py              # PDF parsing, chunking, retrieval, and LLM prompting
|-- requirements.txt
|-- templates/
|   `-- index.html      # Main app UI
|-- static/
|   |-- style.css       # Responsive product styling
|   `-- script.js       # Upload, document, and chat interactions
|-- uploads/            # Uploaded PDFs
`-- vectorstore/        # Persisted document indexes
```

## Setup

```bash
cd pdf_rag_chatbot
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional, for generated answers:

```bash
set ANTHROPIC_API_KEY=sk-ant-your-key
```

Run the app:

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

## Notes

- TF-IDF is simple, fast, and local. For larger or more semantic search, replace
  the retrieval layer with embeddings plus FAISS, Chroma, or another vector store.
- Scanned PDFs usually require OCR before text can be extracted.
- For deployment, use a production WSGI server, authentication, persistent object
  storage, and a database-backed document registry.
