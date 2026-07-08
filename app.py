import os

import streamlit as st

from backend.rag.vector_store import process_documents
from backend.rag.rag_pipeline import ask_question

# ----------------------------
# Page Configuration
# ----------------------------

st.set_page_config(
    page_title="Medical RAG Chatbot",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Medical RAG Chatbot")

st.write(
    "Upload one or more medical PDF documents and ask questions based on them."
)

# ----------------------------
# Session State
# ----------------------------

if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

# ----------------------------
# Upload Folder
# ----------------------------

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----------------------------
# Upload PDFs
# ----------------------------

uploaded_files = st.file_uploader(
    "📂 Upload Medical PDF(s)",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    st.success(f"{len(uploaded_files)} PDF(s) selected.")

    for file in uploaded_files:
        st.write(f"📄 {file.name}")

    if st.button("📥 Process Documents"):

        with st.spinner("Processing documents..."):

            # Remove previously uploaded PDFs
            for filename in os.listdir(UPLOAD_FOLDER):

                file_path = os.path.join(
                    UPLOAD_FOLDER,
                    filename
                )

                if os.path.isfile(file_path):
                    os.remove(file_path)

            # Save newly uploaded PDFs
            for uploaded_file in uploaded_files:

                save_path = os.path.join(
                    UPLOAD_FOLDER,
                    uploaded_file.name
                )

                with open(save_path, "wb") as file:
                    file.write(uploaded_file.getbuffer())

            process_documents(UPLOAD_FOLDER)

            st.session_state.documents_processed = True

        st.success("✅ Documents processed successfully!")

# ----------------------------
# Ask Questions
# ----------------------------
st.write("documents_processed =", st.session_state.documents_processed)
if st.session_state.documents_processed:

    st.divider()

    st.subheader("Ask Questions")

    question = st.text_input(
        "Enter your question:"
    )

    if st.button("Ask"):

        if not question.strip():
            st.warning("Please enter a question.")

        else:

            with st.spinner("Generating answer..."):

                result = ask_question(question)

            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Source Document(s)")

            for source in result["sources"]:
                st.write(f"📄 {source}")

            with st.expander("View Retrieved Context"):
                st.write(result["context"])