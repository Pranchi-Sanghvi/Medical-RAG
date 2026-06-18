import chromadb
from sentence_transformers import SentenceTransformer

# Connect to ChromaDB
client = chromadb.PersistentClient(
    path="backend/rag/chroma_db"
)

collection = client.get_collection(
    name="diabetes_docs"
)

# Load embedding model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# User question
query = input("Ask a question: ")

# Convert question to embedding
query_embedding = model.encode(query)

# Retrieve top chunks
results = collection.query(
    query_embeddings=[
        query_embedding.tolist()
    ],
    n_results=3
)

print("\nRetrieved Context:\n")

for i, doc in enumerate(results["documents"][0], start=1):
    print(f"\nChunk {i}:")
    print(doc)