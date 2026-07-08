import os
import chromadb
from backend.rag.pdf_loader import load_pdf
from backend.rag.chunker import create_chunks
from backend.embeddings.embedder import generate_embeddings


# ----------------------------
# ChromaDB
# ----------------------------

client = chromadb.PersistentClient(
    path="backend/rag/chroma_db"
)


# ----------------------------
# Process Documents
# ----------------------------

def process_documents(upload_folder):

    collection_name = "medical_docs"

    # Delete old collection if it exists
    try:
        client.delete_collection(collection_name)
    except:
        pass

    collection = client.get_or_create_collection(
        name=collection_name
    )

    document_id = 0

    # Loop through every uploaded PDF
    for filename in os.listdir(upload_folder):

        if not filename.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(upload_folder, filename)

        print(f"Reading {filename}")

        text = load_pdf(pdf_path)

        chunks = create_chunks(text)

        embeddings = generate_embeddings(chunks)

        ids = []
        documents = []
        vectors = []
        metadatas = []

        for i, chunk in enumerate(chunks):

            ids.append(str(document_id))

            documents.append(chunk)

            vectors.append(
                embeddings[i].tolist()
            )

            metadatas.append(
                {
                    "source": filename,
                    "chunk": i + 1
                }
            )

            document_id += 1

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=vectors,
            metadatas=metadatas
        )

        print(f"{filename} stored successfully.")

    print("\nAll documents processed successfully!")

