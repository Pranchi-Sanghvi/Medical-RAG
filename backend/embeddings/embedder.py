from sentence_transformers import SentenceTransformer

# Load embedding model only once
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def generate_embeddings(chunks):
    """
    Generate embeddings for a list of text chunks.
    """

    embeddings = model.encode(chunks)

    return embeddings

