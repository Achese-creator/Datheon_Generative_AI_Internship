import os

from dotenv import load_dotenv
from google import genai

load_dotenv()


def generate_answer(prompt: str) -> str:
    """Generate a text answer using the configured Gemini model."""
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Configure it in your .env file."
        )

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        contents=prompt,
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    return response.text
