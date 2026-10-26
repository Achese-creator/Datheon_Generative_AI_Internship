from collections.abc import Callable

from pydantic import ValidationError

from task1_structured_extractor.models import ResumeProfile


class ExtractionError(Exception):
    """Raised when an LLM response cannot be validated."""


def build_prompt(resume_text: str) -> str:
    """Create instructions for extracting resume information as JSON."""
    return f"""
Extract the information from the resume below.

Return ONLY a valid JSON object with these fields:
- full_name: string, required
- email: string or null
- phone: string or null
- skills: list of strings
- experience: list of objects with role, company, duration, achievements
- education: list of objects with qualification, institution, graduation_year

Use null for unknown optional scalar values.
Use empty lists when no skills, experience, or education are provided.
Do not invent information that is not present in the resume.
For graduation_year, use an integer or null.

RESUME:
{resume_text}
""".strip()


def validate_response(json_text: str) -> ResumeProfile:
    """Validate a JSON string against the expected resume schema."""
    try:
        return ResumeProfile.model_validate_json(json_text)
    except (ValidationError, ValueError) as exc:
        raise ExtractionError(
            f"The response was not valid resume JSON: {exc}"
        ) from exc


def extract_resume(
    resume_text: str,
    llm_call: Callable[[str], str],
) -> ResumeProfile:
    """
    Ask an injected LLM function to extract a resume, then validate its JSON.

    llm_call accepts a prompt and returns the model's raw text response.
    Injecting the model call makes this pipeline testable without API costs.
    """
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text cannot be empty.")

    prompt = build_prompt(resume_text)
    response = llm_call(prompt)

    if not isinstance(response, str) or not response.strip():
        raise ExtractionError("The model returned an empty response.")

    return validate_response(response)
