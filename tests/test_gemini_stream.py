import os
from unittest.mock import MagicMock

import pytest

from task4_streamlit_chatbot.gemini_stream import stream_gemini_answer


def test_streams_text_from_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    fake_client = MagicMock()
    fake_client.models.generate_content_stream.return_value = [
        MagicMock(text="First "),
        MagicMock(text="second"),
        MagicMock(text=None),
    ]

    monkeypatch.setattr(
        "task4_streamlit_chatbot.gemini_stream.genai.Client",
        lambda **kwargs: fake_client,
    )

    result = list(stream_gemini_answer("Test prompt"))

    assert result == ["First ", "second"]
    fake_client.models.generate_content_stream.assert_called_once()


def test_missing_api_key_raises_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        list(stream_gemini_answer("Test prompt"))


def test_uses_configured_model(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")

    fake_client = MagicMock()
    fake_client.models.generate_content_stream.return_value = [
        MagicMock(text="Hello")
    ]

    monkeypatch.setattr(
        "task4_streamlit_chatbot.gemini_stream.genai.Client",
        lambda **kwargs: fake_client,
    )

    assert list(stream_gemini_answer("Test prompt")) == ["Hello"]

    call_kwargs = fake_client.models.generate_content_stream.call_args.kwargs
    assert call_kwargs["model"] == "test-model"


def test_empty_stream_returns_no_text(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    fake_client = MagicMock()
    fake_client.models.generate_content_stream.return_value = [
        MagicMock(text=None),
        MagicMock(text=""),
    ]

    monkeypatch.setattr(
        "task4_streamlit_chatbot.gemini_stream.genai.Client",
        lambda **kwargs: fake_client,
    )

    assert list(stream_gemini_answer("Test prompt")) == []
