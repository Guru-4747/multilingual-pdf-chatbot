import os
import re
import hashlib

import faiss
import numpy as np
import streamlit as st

from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Multilingual AI PDF Chatbot",
    page_icon="📚",
    layout="wide"
)

load_dotenv()


# =========================================================
# API KEY
# =========================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Gemini API key not found.")
    st.info("Create a .env file and add GEMINI_API_KEY=YOUR_KEY")
    st.stop()

client = genai.Client(api_key=api_key)


# =========================================================
# MULTILINGUAL EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


embedding_model = load_embedding_model()


# =========================================================
# LANGUAGES
# =========================================================

LANGUAGES = {
    "English": "English",
    "தமிழ் (Tamil)": "Tamil",
    "ಕನ್ನಡ (Kannada)": "Kannada",
    "हिन्दी (Hindi)": "Hindi",
    "తెలుగు (Telugu)": "Telugu",
    "മലയാളം (Malayalam)": "Malayalam"
}


# =========================================================
# SESSION STATE
# =========================================================

if "processed" not in st.session_state:
    st.session_state.processed = False

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# TITLE
# =========================================================

st.title("📚 Multilingual AI PDF Chatbot")

st.write(
    "Upload a PDF, ask questions, and receive answers "
    "in English, Tamil, Kannada, Hindi, Telugu, or Malayalam."
)

st.caption(
    "Designed for large PDFs, including 1000+ pages, "
    "subject to file size and available system resources."
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Settings")

    selected_language = st.selectbox(
        "Answer Language",
        list(LANGUAGES.keys())
    )

    top_k = st.slider(
        "Relevant chunks",
        min_value=2,
        max_value=8,
        value=4
    )

    st.divider()

    st.write("### Supported Languages")

    st.write("🇬🇧 English")
    st.write("🇮🇳 தமிழ்")
    st.write("🇮🇳 ಕನ್ನಡ")
    st.write("🇮🇳 हिन्दी")
    st.write("🇮🇳 తెలుగు")
    st.write("🇮🇳 മലയാളം")


# =========================================================
# PDF UPLOADER
# =========================================================

uploaded_file = st.file_uploader(
    "📄 Upload your PDF",
    type=["pdf"],
    max_upload_size=1024
)


# =========================================================
# HELPER: FILE HASH
# =========================================================

def get_file_hash(uploaded_file):

    file_bytes = uploaded_file.getvalue()

    return hashlib.md5(file_bytes).hexdigest()


# =========================================================
# EXTRACT PDF PAGE BY PAGE
# =========================================================

def extract_pdf_chunks(uploaded_file):

    reader = PdfReader(uploaded_file)

    total_pages = len(reader.pages)

    chunks = []

    progress = st.progress(0)

    status = st.empty()

    for page_number, page in enumerate(reader.pages, start=1):

        status.write(
            f"Reading page {page_number} of {total_pages}..."
        )

        try:
            text = page.extract_text() or ""

        except Exception:
            text = ""

        text = clean_text(text)

        if text:

            page_chunks = create_chunks(
                text,
                page_number
            )

            chunks.extend(page_chunks)

        progress.progress(
            page_number / total_pages
        )

    status.success(
        f"Finished reading {total_pages} pages."
    )

    progress.empty()

    return chunks, total_pages


# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):

    text = text.replace("\x00", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# CREATE CHUNKS
# =========================================================

def create_chunks(
    text,
    page_number,
    chunk_size=700,
    overlap=100
):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = min(
            start + chunk_size,
            len(words)
        )

        chunk_text = " ".join(
            words[start:end]
        )

        if chunk_text.strip():

            chunks.append({
                "text": chunk_text,
                "page": page_number
            })

        if end == len(words):
            break

        start = end - overlap

    return chunks


# =========================================================
# CREATE EMBEDDINGS
# =========================================================

def create_embeddings(chunks):

    texts = [
        item["text"]
        for item in chunks
    ]

    embeddings = []

    batch_size = 32

    progress = st.progress(0)

    total = len(texts)

    for start in range(
        0,
        total,
        batch_size
    ):

        batch = texts[
            start:start + batch_size
        ]

        batch_embeddings = embedding_model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        embeddings.extend(
            batch_embeddings
        )

        progress.progress(
            min(
                (start + batch_size) / total,
                1.0
            )
        )

    progress.empty()

    return np.asarray(
        embeddings,
        dtype="float32"
    )


# =========================================================
# CREATE FAISS INDEX
# =========================================================

def create_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return index


# =========================================================
# PROCESS PDF
# =========================================================

if uploaded_file:

    current_hash = get_file_hash(
        uploaded_file
    )

    file_changed = (
        current_hash !=
        st.session_state.file_hash
    )

    if file_changed:

        st.session_state.processed = False

        st.session_state.chunks = []

        st.session_state.index = None

        st.session_state.messages = []

        st.session_state.file_hash = current_hash

    st.info(
        f"📄 File: {uploaded_file.name}"
    )

    if st.button(
        "🚀 Process PDF",
        type="primary"
    ):

        with st.spinner(
            "Processing PDF. Large PDFs may take some time..."
        ):

            chunks, total_pages = extract_pdf_chunks(
                uploaded_file
            )

            if not chunks:

                st.error(
                    "No readable text was found."
                )

                st.warning(
                    "This may be a scanned/image-only PDF. "
                    "OCR support is required for scanned PDFs."
                )

                st.stop()

            st.write(
                f"Created {len(chunks)} text chunks "
                f"from {total_pages} pages."
            )

            embeddings = create_embeddings(
                chunks
            )

            index = create_faiss_index(
                embeddings
            )

            st.session_state.chunks = chunks

            st.session_state.index = index

            st.session_state.processed = True

            st.success(
                "✅ PDF processed successfully!"
            )


# =========================================================
# SEARCH PDF
# =========================================================

def search_pdf(
    question,
    chunks,
    index,
    top_k
):

    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32"
    )

    scores, positions = index.search(
        question_embedding,
        top_k
    )

    results = []

    for score, position in zip(
        scores[0],
        positions[0]
    ):

        if position == -1:
            continue

        results.append({
            "text": chunks[position]["text"],
            "page": chunks[position]["page"],
            "score": float(score)
        })

    return results


# =========================================================
# GENERATE AI ANSWER
# =========================================================

def generate_answer(
    question,
    results,
    language
):

    context_parts = []

    for result in results:

        context_parts.append(
            f"[Page {result['page']}]\n"
            f"{result['text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
You are a multilingual PDF question-answering assistant.

Answer the user's question using ONLY the information
provided in the PDF context.

Target answer language:
{language}

Rules:

1. Answer in {language}.
2. Do not invent information.
3. If the answer is not present in the context,
   say that the information could not be found
   in the uploaded PDF.
4. Keep the answer clear and easy to understand.
5. Preserve important technical terms when necessary.
6. Mention the relevant PDF page number(s) at the end.
7. If the user asks for an explanation, explain simply.
8. Do not use knowledge outside the provided context.

PDF CONTEXT:

{context}

USER QUESTION:

{question}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text


# =========================================================
# CHAT HISTORY
# =========================================================

if st.session_state.messages:

    st.subheader("💬 Chat")

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


# =========================================================
# QUESTION INPUT
# =========================================================

if st.session_state.processed:

    question = st.chat_input(
        "Ask a question about your PDF..."
    )

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching the PDF..."
            ):

                results = search_pdf(
                    question,
                    st.session_state.chunks,
                    st.session_state.index,
                    top_k
                )

            with st.spinner(
                "Generating answer..."
            ):

                answer = generate_answer(
                    question,
                    results,
                    LANGUAGES[selected_language]
                )

            st.markdown(answer)

            st.divider()

            st.caption("📑 Sources")

            pages = sorted(
                set(
                    result["page"]
                    for result in results
                )
            )

            st.write(
                "Relevant pages:",
                ", ".join(
                    str(page)
                    for page in pages
                )
            )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })


# =========================================================
# EXTRA FEATURES
# =========================================================

if st.session_state.processed:

    st.divider()

    st.subheader("✨ Extra Tools")

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "📝 Generate Summary"
        ):

            results = search_pdf(
                "Give an overall summary of this document",
                st.session_state.chunks,
                st.session_state.index,
                top_k=8
            )

            summary = generate_answer(
                "Give an overall summary of the available document content.",
                results,
                LANGUAGES[selected_language]
            )

            st.write(summary)

    with col2:

        if st.button(
            "❓ Generate MCQs"
        ):

            results = search_pdf(
                "Important concepts and facts in this document",
                st.session_state.chunks,
                st.session_state.index,
                top_k=8
            )

            mcqs = generate_answer(
                "Create 5 multiple-choice questions from the retrieved PDF content. "
                "Give four options and identify the correct answer.",
                results,
                LANGUAGES[selected_language]
            )

            st.write(mcqs)