"""
app.py
------
Flask web server for the AI PDF Chatbot (RAG).

Endpoints:
  GET  /                  -> chat UI
  POST /upload             -> upload a PDF, process & index it
  GET  /documents          -> list indexed documents
  POST /chat                -> ask a question about a document

Run:
  pip install -r requirements.txt
  export ANTHROPIC_API_KEY=sk-ant-...
  python app.py
Then open http://localhost:5000
"""

import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import anthropic
import rag

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
rag.load_persisted_documents()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB max upload


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    try:
        doc = rag.process_pdf(filepath, filename)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to process PDF: {e}"}), 500

    return jsonify({
        "doc_id": doc.doc_id,
        "filename": doc.filename,
        "chunks": len(doc.chunks),
        "message": f"Indexed '{doc.filename}' into {len(doc.chunks)} chunks."
    })


@app.route("/documents", methods=["GET"])
def documents():
    return jsonify(rag.list_documents())


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    doc_id = data.get("doc_id")
    question = (data.get("question") or "").strip()

    if not doc_id:
        return jsonify({"error": "doc_id is required"}), 400
    if not question:
        return jsonify({"error": "question is required"}), 400

    doc = rag.get_document(doc_id)
    if doc is None:
        return jsonify({"error": "Document not found. Upload it again."}), 404

    retrieved = rag.retrieve(doc, question)
    if not retrieved:
        return jsonify({
            "answer": "I couldn't find anything relevant to that question in the document.",
            "sources": []
        })

    try:
        answer = rag.ask_claude(question, retrieved)
    except RuntimeError as e:
        # No API key set -> fall back to showing raw retrieved context so the
        # retrieval half of the project is still demonstrable end-to-end.
        fallback = "\n\n".join(f"[Excerpt {i+1}] {r['chunk'][:300]}..." for i, r in enumerate(retrieved))
        return jsonify({
            "answer": f"(LLM not configured: {e})\n\nMost relevant excerpts found:\n\n{fallback}",
            "sources": [{"score": r["score"]} for r in retrieved],
            "warning": str(e)
        })

    return jsonify({
        "answer": answer,
        "sources": [{"score": round(r["score"], 3), "preview": r["chunk"][:150] + "..."} for r in retrieved]
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
