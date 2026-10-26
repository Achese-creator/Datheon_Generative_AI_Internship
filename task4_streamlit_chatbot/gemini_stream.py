import os

from google import genai


def stream_gemini_answer(prompt: str):
    """Yield generated text chunks from Gemini; never expose internal reasoning."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    client = genai.Client(api_key=api_key)
    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=prompt,
    ):
        text = getattr(chunk, "text", None)
        if text:
            yield text
