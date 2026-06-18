from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz

# Load PDF

pdf_path = "../../data/pdfs/Diabetes.pdf"  # can also write :- pdf_path = r"C:\Users\PRANCHI\medical-rag\data\pdfs\Diabetes.pdf"

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

print("Number of chunks:", len(chunks))

print("\nFirst chunk:\n")
print(chunks[0])