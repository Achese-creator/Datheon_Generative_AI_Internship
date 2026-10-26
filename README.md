# Datheon Generative AI Internship

Python projects for the Datheon Python + Generative AI & LangChain internship.

## Task 1: Structured LLM Prompt Pipeline & JSON Extractor

Extracts resume information into a Pydantic-validated JSON schema.

### Features

- Structured resume schema with Pydantic.
- Prompt construction for LLM-based extraction.
- JSON response validation and clear errors.
- Demo mode for testing without API requests.
- Optional Gemini API integration.
- Automated tests using pytest.

### Requirements

- Python 3.11+
- pip

### Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Run the demo

```powershell
python -m task1_structured_extractor.cli --mode demo
```

The demo uses a predefined sample response. It does not call an LLM.

### Run tests

```powershell
python -m pytest -v
```

### Run with Gemini

1. Copy `.env.example` to `.env`.
2. Add a valid Gemini API key to `.env`.
3. Confirm your selected model is available to your account.
4. Run:

```powershell
python -m task1_structured_extractor.cli --mode live
```

Live mode may consume API quota or incur charges depending on your account and provider terms.

### Project status

Task 1 is under development. The demo validates a fixed JSON response; live extraction requires a configured Gemini API key.

## Planned tasks

- Task 2: PDF question answering with RAG and source citations.
- Task 3: Multi-tool AI agent.
- Task 4: Streamlit conversational interface.

## Security

Never commit `.env`, API keys, private documents, or other secrets.
