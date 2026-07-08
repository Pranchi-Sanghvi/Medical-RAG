# 🩺 Medical RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers medical questions from user-uploaded PDF documents.

The application allows users to upload one or more medical PDFs, processes them into vector embeddings using Sentence Transformers, stores them in ChromaDB, retrieves the most relevant information based on semantic similarity, and generates accurate answers using a Large Language Model (LLM).

---

## 🚀 Live Demo

https://medical-rag-pranchisanghvi.streamlit.app/

---

## ✨ Features

- Upload one or multiple PDF documents
- Automatic PDF text extraction
- Intelligent text chunking
- Sentence Transformer embeddings
- ChromaDB vector database
- Semantic document retrieval
- LLM-powered answer generation
- Source document identification
- Reduced hallucinations using retrieved context
- Interactive Streamlit interface

---

## 🏗️ Project Architecture

```text
                User Uploads PDFs
                        │
                        ▼
                 PDF Text Extraction
                        │
                        ▼
                  Text Chunking
                        │
                        ▼
             Sentence Embeddings
                        │
                        ▼
                 ChromaDB Storage
                        │
                        ▼
                  User Question
                        │
                        ▼
               Semantic Retrieval
                        │
                        ▼
                Retrieved Context
                        │
                        ▼
            Hugging Face LLM (Qwen)
                        │
                        ▼
                  Final Answer
```

---

## 🛠️ Technologies Used

- Python
- Streamlit
- ChromaDB
- Sentence Transformers
- Hugging Face Inference API
- LangChain Text Splitter
- PyMuPDF (fitz)

---

## 📂 Project Structure

```text
medical-rag/
│
├── app.py
├── requirements.txt
├── README.md
│
├── backend/
│   ├── embeddings/
│   │   └── embedder.py
│   │
│   └── rag/
│       ├── pdf_loader.py
│       ├── chunker.py
│       ├── vector_store.py
│       └── rag_pipeline.py
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/Pranchi-Sanghvi/Medical-RAG.git
```

Move into the project

```bash
cd Medical-RAG
```

Create virtual environment

```bash
python -m venv venv
```

Activate virtual environment

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file.

```text
HF_TOKEN=your_huggingface_token
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 📖 How to Use

1. Launch the application.
2. Upload one or more medical PDF files.
3. Click **Process Documents**.
4. Ask questions related to the uploaded documents.
5. View the generated answer, retrieved context, and source document names.

---

## 🔮 Future Improvements

- Chat history
- User authentication
- Conversation memory
- Hybrid search (keyword + semantic)
- Citation highlighting
- Support for DOCX and TXT files
- Docker deployment

---

## 👩‍💻 Author

**Pranchi Sanghvi**

GitHub: https://github.com/Pranchi-Sanghvi