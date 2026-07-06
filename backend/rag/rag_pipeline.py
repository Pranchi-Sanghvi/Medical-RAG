import chromadb
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()

# ChromaDB

client = chromadb.PersistentClient(
    path="backend/rag/chroma_db"
)

collection = client.get_collection(
    name="diabetes_docs"
)

# Embedding Model

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# LLM

llm = InferenceClient(
    token=os.getenv("HF_TOKEN")
)

# Ask Question

def ask_question(question):

    query_embedding = embedding_model.encode(
        question
    )

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=3
    )

    context = "\n".join(
        results["documents"][0]
    )

    prompt = f"""
Answer ONLY using the context below.

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

    return {
        "question": question,
        "context": context,
        "answer": response.choices[0].message.content
    }


if __name__ == "__main__":

    question = input("Enter your question: ")

    result = ask_question(question)

    print("\nRETRIEVED CONTEXT:")
    print(result["context"])

    print("\nQUESTION:")
    print(result["question"])

    print("\nANSWER:")
    print(result["answer"])