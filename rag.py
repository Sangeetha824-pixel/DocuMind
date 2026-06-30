"""
rag.py
------
Core Retrieval-Augmented-Generation engine for the PDF Chatbot.

Pipeline:
  1. Extract text from an uploaded PDF (pypdf)
  2. Split text into overlapping chunks
  3. Vectorize chunks with TF-IDF (no external embedding API needed -> works offline)
  4. On a user question, retrieve the top-k most similar chunks (cosine similarity)
  5. Send the question + retrieved chunks to Claude as context and stream back an answer

This keeps the project fully self-contained (no vector DB server, no embedding API key
required for retrieval) while still using a real LLM (Claude) for the final answer.
"""

import os
import re
import pickle
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import anthropic
except ImportError:  # anthropic is optional until generation is actually used
    anthropic = None


CHUNK_SIZE = 1000        # characters per chunk
CHUNK_OVERLAP = 150      # overlap between consecutive chunks
TOP_K = 4                # number of chunks to retrieve per question

VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "vectorstore")
os.makedirs(VECTORSTORE_DIR, exist_ok=True)


@dataclass
class Document:
    doc_id: str
    filename: str
    chunks: List[str] = field(default_factory=list)
    vectorizer: Optional[TfidfVectorizer] = None
    matrix: object = None  # sparse TF-IDF matrix, one row per chunk


# In-memory registry of processed documents for this server process.
# Persisted to disk too, so a restart can reload without re-uploading.
_DOCUMENTS: Dict[str, Document] = {}


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_from_pdf(filepath: str) -> str:
    reader = PdfReader(filepath)
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = _clean_text(text)
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def process_pdf(filepath: str, filename: str) -> Document:
    """Extract, chunk, and vectorize a PDF. Returns the Document and stores it in memory."""
    raw_text = extract_text_from_pdf(filepath)
    chunks = chunk_text(raw_text)

    if not chunks:
        raise ValueError("No extractable text found in this PDF (it may be a scanned image).")

    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
    matrix = vectorizer.fit_transform(chunks)

    doc_id = str(uuid.uuid4())[:8]
    doc = Document(doc_id=doc_id, filename=filename, chunks=chunks, vectorizer=vectorizer, matrix=matrix)
    _DOCUMENTS[doc_id] = doc

    # Persist to disk so chunks/vectorizer survive a server restart
    with open(os.path.join(VECTORSTORE_DIR, f"{doc_id}.pkl"), "wb") as f:
        pickle.dump(doc, f)

    return doc


def list_documents() -> List[Dict]:
    return [{"doc_id": d.doc_id, "filename": d.filename, "chunks": len(d.chunks)} for d in _DOCUMENTS.values()]


def load_persisted_documents() -> None:
    """Load previously indexed documents from disk into the in-memory registry."""
    for filename in os.listdir(VECTORSTORE_DIR):
        if not filename.endswith(".pkl"):
            continue

        path = os.path.join(VECTORSTORE_DIR, filename)
        try:
            with open(path, "rb") as f:
                doc = pickle.load(f)
            _DOCUMENTS[doc.doc_id] = doc
        except Exception:
            # A corrupt index should not prevent the app from starting.
            continue


def get_document(doc_id: str) -> Optional[Document]:
    if doc_id in _DOCUMENTS:
        return _DOCUMENTS[doc_id]
    # try loading from disk
    path = os.path.join(VECTORSTORE_DIR, f"{doc_id}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            doc = pickle.load(f)
        _DOCUMENTS[doc_id] = doc
        return doc
    return None


def retrieve(doc: Document, question: str, top_k: int = TOP_K) -> List[Dict]:
    """Return the top_k most relevant chunks for the question, with similarity scores."""
    q_vec = doc.vectorizer.transform([question])
    sims = cosine_similarity(q_vec, doc.matrix).flatten()
    top_idx = np.argsort(sims)[::-1][:top_k]
    results = []
    for idx in top_idx:
        if sims[idx] <= 0:
            continue
        results.append({"chunk": doc.chunks[idx], "score": float(sims[idx]), "index": int(idx)})
    return results


def build_prompt(question: str, retrieved: List[Dict]) -> str:
    context = "\n\n---\n\n".join(r["chunk"] for r in retrieved)
    prompt = f"""You are a helpful assistant answering questions about a PDF document.
Use ONLY the context excerpts below to answer. If the answer isn't in the context, say you don't know based on the document.

Context excerpts:
{context}

Question: {question}

Answer clearly and concisely, citing relevant details from the context where useful."""
    return prompt


def ask_claude(question: str, retrieved: List[Dict], api_key: Optional[str] = None) -> str:
    """Generate an answer using Claude, grounded in the retrieved chunks."""
    if anthropic is None:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Set the ANTHROPIC_API_KEY environment variable to enable answers.")

    client = anthropic.Anthropic(api_key=api_key)
    prompt = build_prompt(question, retrieved)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")
