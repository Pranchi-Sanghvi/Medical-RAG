import fitz
import chromadb

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDF

pdf_path = "../../data/pdfs/Diabetes.pdf" # or pdf_path = r"C:\Users\PRANCHI\medical-rag\data\pdfs\Diabetes.pdf"

doc = fitz.open(pdf_path)

text = ""

for page in doc:
    text += page.get_text()

# Create chunks

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

chunks = splitter.split_text(text)

# Embeddings

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = model.encode(chunks)

# ChromaDB

client = chromadb.PersistentClient(
    path="backend/rag/chroma_db"
)

collection = client.get_or_create_collection(
    name="diabetes_docs"
)

# Store

ids = [str(i) for i in range(len(chunks))]

collection.add(
    ids=ids,
    documents=chunks,
    embeddings=embeddings.tolist()
)

print("Stored Successfully!")