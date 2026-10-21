import os
import re

import faiss
import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

# ---------- Config ----------
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 900
OVERLAP = 150
TOP_K = 4
MIN_SCORE = 0.25
HISTORY_LIMIT = 8

GENERAL_PROMPT = "You are KnowBase AI, a helpful and concise assistant."
RAG_PROMPT = (
    "You are KnowBase AI. Answer the user's question using ONLY the document "
    "excerpts provided below. Each excerpt is labelled with its source and page. "
    "If the excerpts do not contain enough information to answer, reply exactly: "
    "\"I couldn't find that in the uploaded document.\" "
    "Do not invent facts. Mention page numbers when helpful."
)
NOT_FOUND = "I couldn't find that in the uploaded document."


# ---------- Helpers ----------
def clean(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found. Add it to your .env file and restart.")
        st.stop()
    return Groq(api_key=api_key)


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)


# ---------- RAG pipeline ----------
def extract_pages(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = [(i + 1, page.get_text().strip()) for i, page in enumerate(doc)]
    return [(num, text) for num, text in pages if text]


def chunk_pages(pages, source):
    chunks = []
    step = CHUNK_SIZE - OVERLAP
    for page_no, text in pages:
        for start in range(0, len(text), step):
            piece = text[start:start + CHUNK_SIZE].strip()
            if len(piece) > 50:
                chunks.append({"text": piece, "source": source, "page": page_no})
    return chunks


def build_index(chunks, embedder):
    vectors = embedder.encode(
        [c["text"] for c in chunks], normalize_embeddings=True
    ).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def retrieve(question, index, chunks, embedder):
    q = embedder.encode([question], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, min(TOP_K, len(chunks)))
    return [
        {**chunks[i], "score": float(s)}
        for s, i in zip(scores[0], ids[0])
        if i != -1 and s >= MIN_SCORE
    ]


def build_messages(question, history, hits, has_doc):
    if has_doc:
        context = "\n\n".join(
            f"[Source: {h['source']}, page {h['page']}]\n{h['text']}" for h in hits
        )
        system = f"{RAG_PROMPT}\n\nDOCUMENT EXCERPTS:\n{context}"
    else:
        system = GENERAL_PROMPT
    return (
        [{"role": "system", "content": system}]
        + history[-HISTORY_LIMIT:]
        + [{"role": "user", "content": question}]
    )


def show_sources(sources):
    with st.expander("Sources"):
        for s in sources:
            st.markdown(f"**{s['source']} - page {s['page']}** (score {s['score']:.2f})")
            st.caption(s["text"][:300] + "...")


# ---------- App ----------
st.set_page_config(page_title="KnowBase AI", page_icon="📚", layout="wide")
st.title("📚 KnowBase AI")
st.caption("Chat with an AI, or upload a PDF and ask questions about it.")

client = get_client()
embedder = load_embedder()

ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("index", None)
ss.setdefault("chunks", [])
ss.setdefault("doc_name", None)

with st.sidebar:
    st.header("Document")
    uploaded = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded and st.button("Process PDF", type="primary", use_container_width=True):
        with st.spinner("Reading and indexing..."):
            pages = extract_pages(uploaded.getvalue())
            chunks = chunk_pages(pages, uploaded.name)
            if not chunks:
                st.error("No extractable text found. Is this a scanned PDF?")
            else:
                ss.index = build_index(chunks, embedder)
                ss.chunks = chunks
                ss.doc_name = uploaded.name
                ss.messages = []
                st.success(f"Indexed {len(pages)} page(s), {len(chunks)} chunks.")

    if ss.doc_name:
        st.info(f"Active document: **{ss.doc_name}**")
        if st.button("Remove document", use_container_width=True):
            ss.index, ss.chunks, ss.doc_name = None, [], None
            ss.messages = []
            st.rerun()

    if st.button("Clear chat", use_container_width=True):
        ss.messages = []
        st.rerun()
    st.caption(f"Model: {MODEL}")

# Show previous messages
for msg in ss.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])

# Handle a new question
question = st.chat_input("Ask a question...")
if question:
    with st.chat_message("user"):
        st.markdown(question)

    history = [{"role": m["role"], "content": m["content"]} for m in ss.messages]
    has_doc = ss.index is not None
    hits = retrieve(question, ss.index, ss.chunks, embedder) if has_doc else []

    with st.chat_message("assistant"):
        if has_doc and not hits:
            answer = NOT_FOUND  # nothing relevant found: skip the API call
        else:
            with st.spinner("Thinking..."):
                try:
                    response = client.chat.completions.create(
                        model=MODEL,
                        messages=build_messages(question, history, hits, has_doc),
                        temperature=0.2,
                        max_tokens=1000,
                    )
                    answer = clean(response.choices[0].message.content)
                except Exception as e:
                    answer = f"Something went wrong: {e}"
        st.markdown(answer)
        if hits:
            show_sources(hits)

    ss.messages.append({"role": "user", "content": question})
    ss.messages.append({"role": "assistant", "content": answer, "sources": hits})