import chromadb
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()

# ----------------------------
# ChromaDB
# ----------------------------

client = chromadb.PersistentClient(
    path="backend/rag/chroma_db"
)

# ----------------------------
# Embedding Model
# ----------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ----------------------------
# LLM
# ----------------------------

llm = InferenceClient(
    token=os.getenv("HF_TOKEN")
)

# ----------------------------
# Ask Question
# ----------------------------

def ask_question(question):

    collection = client.get_collection(
        name="medical_docs"
    )

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

    response = llm.chat_completion(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=200
    )

    # Remove duplicate filenames
    sources = sorted(
        list(
            {
                metadata["source"]
                for metadata in metadatas
            }
        )
    )

    return {
        "question": question,
        "context": context,
        "answer": response.choices[0].message.content,
        "sources": sources
    }

