import fitz
import os


def load_pdf(pdf_path):
    """
    Reads a PDF and returns the extracted text.
    """

    doc = fitz.open(pdf_path)

    text = ""

    for page in doc:
        text += page.get_text()

    doc.close()

    return text

