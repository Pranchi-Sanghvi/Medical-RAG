import fitz
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDF

pdf_path = "../../data/pdfs/Diabetes.pdf"  # or pdf_path = r"C:\Users\PRANCHI\medical-rag\data\pdfs\Diabetes.pdf"

doc = fitz.open(pdf_path)

text = ""

for page in doc:
    text += page.get_text()

# Chunking

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

chunks = splitter.split_text(text)

# Load embedding model

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# Generate embeddings

embeddings = model.encode(chunks)

print("Embedding shape:")
print(embeddings.shape)