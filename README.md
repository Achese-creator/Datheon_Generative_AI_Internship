# Datheon Generative AI Internship

Python projects for the Datheon Python + Generative AI & LangChain internship.

## Task 1: Structured LLM Prompt Pipeline & JSON Extractor

Extracts resume information into a Pydantic-validated JSON schema.

### Features
- Structured resume schema with Pydantic
- Prompt construction for LLM-based extraction
- JSON response validation and clear errors
- Demo mode without API requests
- Optional Gemini API integration
- Automated tests using pytest

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

### Run tests
```powershell
python -m pytest -v
```

### Run with Gemini
1. Copy `.env.example` to `.env`.
2. Add your Gemini API key to `.env`.
3. Confirm the configured model is available to your account.
4. Run:

```powershell
python -m task1_structured_extractor.cli --mode live
```

Live mode may use API quota or incur charges, depending on your provider terms.

## Task 2: Document Q&A RAG Assistant

A retrieval-augmented generation (RAG) pipeline for PDF and TXT documents. It loads documents, splits their text into overlapping chunks, creates local embeddings, retrieves relevant passages with ChromaDB, and prepares source-grounded answers.

### Features
- PDF and TXT document loading
- Recursive text chunking with overlap
- Local embeddings using FastEmbed
- ChromaDB vector storage and semantic retrieval
- Source metadata and PDF page references
- Gemini-powered answer generation in live mode
- Demo mode without an API key

### Run the demo
```powershell
python -m task2_rag_assistant.cli --mode demo
```

### Run with Gemini
Configure `GEMINI_API_KEY` in your local `.env` file, then run:

```powershell
python -m task2_rag_assistant.cli --mode live --question "Why is chunk overlap useful?"
```

Demo mode uses a fixed sample response and does not verify answer relevance. Live mode calls the configured Gemini model. Scanned PDFs may require OCR.

## Task 4: Streamlit Conversational Chatbot

A Streamlit chat interface that connects to the Task 2 RAG pipeline.

### Features
- Streamlit chat interface with conversation history
- Session-state memory across chat interactions
- Incremental response display
- Demo mode that avoids LLM API calls
- Optional Gemini streaming generation in live mode
- Retrieved source labels displayed with answers
- Automated tests for the Gemini streaming provider

### Run the chatbot

```powershell
python -m streamlit run .\task4_streamlit_chatbot\app.py
```

Select **Demo (no API calls)** to test without an LLM request. The initial version indexes the included sample document.

For live Gemini responses, configure `GEMINI_API_KEY` in your local `.env` file and select **Live Gemini** in the sidebar. Live requests may consume API quota or incur charges, depending on your provider terms.

## Test suite

Run all automated tests:

```powershell
python -m pytest -v
```

The tests use mocked LLM calls where appropriate, so the test suite does not need a live Gemini API request.

## Remaining task
- Task 3: Multi-tool AI agent

## Security
Never commit `.env`, API keys, private documents, or other secrets.
