import os

from dotenv import load_dotenv
from google import genai

from task1_structured_extractor.models import ResumeProfile

load_dotenv()


def generate_resume_json(prompt: str) -> str:
    """Generate structured resume JSON using the Gemini API."""
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to your .env file."
        )

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": ResumeProfile,
        },
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    return response.text

