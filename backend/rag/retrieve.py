import chromadb
from sentence_transformers import SentenceTransformer

# Connect to existing ChromaDB
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
query = "What are the symptoms of diabetes?"

# Convert question into embedding
query_embedding = model.encode(query)

# Search database
results = collection.query(
    query_embeddings=[
        query_embedding.tolist()
    ],
    n_results=3
)

print("\nTop Results:\n")

for doc in results["documents"][0]:
    print("-" * 50)
    print(doc)