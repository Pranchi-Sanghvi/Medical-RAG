import streamlit as st
from backend.rag.rag_pipeline import ask_question

st.set_page_config(
    page_title="Medical RAG Chatbot",
    page_icon="🩺"
)

st.title("🩺 Medical RAG Chatbot")

st.write("Ask questions based on the uploaded medical documents.")

question = st.text_input(
    "Enter your question:"
)

if st.button("Ask"):

    if question.strip() == "":
        st.warning("Please enter a question.")
    else:

        with st.spinner("Generating answer..."):

            result = ask_question(question)

        st.subheader("Retrieved Context")
        st.write(result["context"])

        st.subheader("Answer")
        st.write(result["answer"])