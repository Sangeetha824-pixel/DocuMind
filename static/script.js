const uploadBtn = document.getElementById("uploadBtn");
const fileInput = document.getElementById("fileInput");
const uploadStatus = document.getElementById("uploadStatus");
const docList = document.getElementById("docList");
const docCount = document.getElementById("docCount");
const activeDocName = document.getElementById("activeDocName");
const modelStatus = document.getElementById("modelStatus");
const messages = document.getElementById("messages");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");

let docs = [];
let activeDocId = "";

uploadBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;

  setUploadStatus("Indexing document. This can take a moment for long PDFs.", "neutral");
  uploadBtn.disabled = true;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/upload", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) {
      setUploadStatus(data.error || "Upload failed.", "error");
      return;
    }

    upsertDocument({
      doc_id: data.doc_id,
      filename: data.filename,
      chunks: data.chunks,
    });
    setActiveDocument(data.doc_id);
    setUploadStatus(data.message, "success");
    clearWelcome();
    addMessage("system", `Ready. "${data.filename}" is indexed into ${data.chunks} chunks.`);
  } catch (err) {
    setUploadStatus(`Upload failed: ${err.message}`, "error");
  } finally {
    uploadBtn.disabled = false;
    fileInput.value = "";
  }
});

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  const question = questionInput.value.trim();
  if (!question) return;

  if (!activeDocId) {
    addMessage("system", "Upload or select a PDF before asking a question.");
    return;
  }

  clearWelcome();
  addMessage("user", question);
  questionInput.value = "";
  setComposerState(false, "Thinking...");

  const thinkingEl = addMessage("bot", "Reading the most relevant passages...");

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ doc_id: activeDocId, question }),
    });
    const data = await res.json();

    thinkingEl.textContent = data.answer || data.error || "No response returned.";

    if (data.warning) {
      modelStatus.textContent = "LLM key missing";
      modelStatus.style.color = "var(--warning)";
      modelStatus.style.background = "#fffbeb";
      modelStatus.style.borderColor = "#fde68a";
    }

    if (Array.isArray(data.sources) && data.sources.length) {
      addSources(data.sources);
    }
  } catch (err) {
    thinkingEl.textContent = `Error: ${err.message}`;
  } finally {
    setComposerState(true, "Ask");
  }
});

async function loadDocuments() {
  try {
    const res = await fetch("/documents");
    const data = await res.json();
    docs = Array.isArray(data) ? data : [];
    renderDocuments();
    if (docs.length > 0) {
      setActiveDocument(docs[0].doc_id, false);
    }
  } catch {
    setUploadStatus("Could not load existing document indexes.", "error");
  }
}

function upsertDocument(doc) {
  docs = [doc, ...docs.filter((item) => item.doc_id !== doc.doc_id)];
  renderDocuments();
}

function renderDocuments() {
  docCount.textContent = docs.length;
  docList.innerHTML = "";

  if (!docs.length) {
    docList.innerHTML = '<div class="empty-note">No documents indexed yet.</div>';
    setComposerState(false, "Ask");
    activeDocName.textContent = "No document selected";
    return;
  }

  docs.forEach((doc) => {
    const button = document.createElement("button");
    button.className = `doc-item${doc.doc_id === activeDocId ? " active" : ""}`;
    button.type = "button";
    button.dataset.docId = doc.doc_id;

    const name = document.createElement("span");
    name.className = "doc-name";
    name.textContent = doc.filename;

    const meta = document.createElement("span");
    meta.className = "doc-meta";
    meta.textContent = `${doc.chunks ?? 0} chunks indexed`;

    button.append(name, meta);
    button.addEventListener("click", () => setActiveDocument(doc.doc_id));
    docList.appendChild(button);
  });
}

function setActiveDocument(docId, announce = true) {
  const doc = docs.find((item) => item.doc_id === docId);
  if (!doc) return;

  activeDocId = doc.doc_id;
  activeDocName.textContent = doc.filename;
  modelStatus.textContent = "Retrieval ready";
  modelStatus.removeAttribute("style");
  setComposerState(true, "Ask");
  renderDocuments();

  if (announce) {
    clearWelcome();
    addMessage("system", `Active document changed to "${doc.filename}".`);
  }
}

function setComposerState(enabled, label) {
  questionInput.disabled = !enabled;
  sendBtn.disabled = !enabled;
  sendBtn.textContent = label;
}

function setUploadStatus(message, type) {
  uploadStatus.textContent = message;
  uploadStatus.className = `status-line${type === "success" ? " success" : ""}${type === "error" ? " error" : ""}`;
}

function clearWelcome() {
  const welcome = messages.querySelector(".welcome-panel");
  if (welcome) welcome.remove();
}

function addMessage(role, text) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.textContent = text;
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
  return el;
}

function addSources(sources) {
  const wrapper = document.createElement("div");
  wrapper.className = "sources";

  const title = document.createElement("div");
  title.className = "sources-title";
  title.textContent = "Retrieved sources";
  wrapper.appendChild(title);

  sources.forEach((source, index) => {
    const item = document.createElement("div");
    item.className = "source-item";

    const score = document.createElement("div");
    score.className = "source-score";
    score.textContent = `Source ${index + 1} - score ${source.score ?? "n/a"}`;
    item.appendChild(score);

    if (source.preview) {
      const preview = document.createElement("div");
      preview.className = "source-preview";
      preview.textContent = source.preview;
      item.appendChild(preview);
    }

    wrapper.appendChild(item);
  });

  messages.appendChild(wrapper);
  messages.scrollTop = messages.scrollHeight;
}

loadDocuments();
