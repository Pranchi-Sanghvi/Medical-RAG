import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-3B-Instruct"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


def generate_answer(context, question):

    prompt = f"""
You are a helpful medical assistant.

Use ONLY the context below to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

    payload = {
        "inputs": prompt
    }

    response = requests.post(
        API_URL,
        headers=headers,
        json=payload
    )

    return response.json()


if __name__ == "__main__":

    context = """
Common symptoms of diabetes include increased thirst,
frequent urination, fatigue, blurred vision, and weight loss.
"""

    question = "What are the symptoms of diabetes?"

    result = generate_answer(
        context,
        question
    )

    print(result)