import fitz

pdf_path = "../../data/pdfs/Diabetes.pdf" # can also write :- pdf_path = r"C:\Users\PRANCHI\medical-rag\data\pdfs\Diabetes.pdf"

doc = fitz.open(pdf_path)

text = ""

for page in doc:
    text += page.get_text()

print("PDF Loaded Successfully!")
print(text[:1000])