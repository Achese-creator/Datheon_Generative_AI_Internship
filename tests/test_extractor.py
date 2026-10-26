import pytest

from task1_structured_extractor.extractor import (
    ExtractionError,
    extract_resume,
    validate_response,
)
from task1_structured_extractor.models import ResumeProfile


VALID_JSON = """
{
  "full_name": "Ada Example",
  "email": "ada@example.com",
  "phone": "+234 800 000 0000",
  "skills": ["Python", "SQL", "data analysis"],
  "experience": [
    {
      "role": "Junior Data Analyst",
      "company": "Example Analytics",
      "duration": "2024-2025",
      "achievements": ["Prepared reports", "Cleaned datasets"]
    }
  ],
  "education": [
    {
      "qualification": "BSc Computer Science",
      "institution": "Example University",
      "graduation_year": 2024
    }
  ]
}
"""


def test_valid_json_returns_resume_profile():
    profile = validate_response(VALID_JSON)

    assert isinstance(profile, ResumeProfile)
    assert profile.full_name == "Ada Example"
    assert profile.skills == ["Python", "SQL", "data analysis"]
    assert profile.education[0].graduation_year == 2024


def test_optional_fields_can_be_null():
    response = """
    {
      "full_name": "Sam Example",
      "email": null,
      "phone": null,
      "skills": [],
      "experience": [],
      "education": []
    }
    """

    profile = validate_response(response)

    assert profile.email is None
    assert profile.experience == []


def test_invalid_json_raises_extraction_error():
    with pytest.raises(ExtractionError):
        validate_response("{not valid JSON}")


def test_missing_required_name_is_rejected():
    response = """
    {
      "skills": [],
      "experience": [],
      "education": []
    }
    """

    with pytest.raises(ExtractionError):
        validate_response(response)


def test_wrong_field_type_is_rejected():
    response = """
    {
      "full_name": "Ada Example",
      "skills": "Python",
      "experience": [],
      "education": []
    }
    """

    with pytest.raises(ExtractionError):
        validate_response(response)


def test_empty_resume_is_rejected_before_model_call():
    def model_must_not_run(prompt):
        raise AssertionError("The model should not be called.")

    with pytest.raises(ValueError, match="cannot be empty"):
        extract_resume("   ", model_must_not_run)


def test_extraction_uses_model_response_and_validates_it():
    def fake_model(prompt):
        assert "Ada Example" in prompt
        assert "valid JSON" in prompt
        return VALID_JSON

    profile = extract_resume(
        "Ada Example. Skills include Python and SQL.",
        fake_model,
    )

    assert profile.full_name == "Ada Example"
    assert "Python" in profile.skills


def test_empty_model_response_is_rejected():
    with pytest.raises(ExtractionError, match="empty response"):
        extract_resume("Some resume text", lambda prompt: "  ")
