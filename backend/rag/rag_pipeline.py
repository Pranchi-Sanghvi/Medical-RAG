import os
import chromadb

from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# Embedding Model
# ----------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ----------------------------
# LLM
# ----------------------------

#llm = InferenceClient(
#    token=os.getenv("HF_TOKEN")
#)
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# ----------------------------
# Ask Question
# ----------------------------

def ask_question(question):

    # Create ChromaDB folder if it doesn't exist
    chroma_path = "backend/rag/chroma_db"
    os.makedirs(chroma_path, exist_ok=True)

    client = chromadb.PersistentClient(
        path=chroma_path
    )

    try:
        collection = client.get_collection(
            name="medical_docs"
        )
    except Exception:

        return {
            "question": question,
            "context": "",
            "answer": "Please upload and process your PDF documents first.",
            "sources": []
        }

    query_embedding = embedding_model.encode(question)

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=3
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context = "\n".join(documents)

    prompt = f"""
Answer ONLY using the context below.

If the answer is not present in the context, reply exactly:

"I couldn't find this information in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:
"""

    #response = llm.chat_completion(
    #    model="Qwen/Qwen2.5-7B-Instruct",
    #    messages=[
    #        {
    #            "role": "user",
    #            "content": prompt
    #        }
    #    ],
    #    max_tokens=200
    #)
    response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    max_tokens=200,
    temperature=0
    )

    sources = sorted(
        {
            metadata["source"]
            for metadata in metadatas
        }
    )

    return {
        "question": question,
        "context": context,
        "answer": response.choices[0].message.content,
        "sources": sources
    }