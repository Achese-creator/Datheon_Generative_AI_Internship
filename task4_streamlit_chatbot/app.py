import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from task2_rag_assistant.chunker import chunk_pages
from task2_rag_assistant.loader import load_document
from task2_rag_assistant.retriever import LocalRetriever
from task2_rag_assistant.answerer import build_answer_prompt
from task4_streamlit_chatbot.gemini_stream import stream_gemini_answer


st.set_page_config(
    page_title="Datheon RAG Chatbot",
    page_icon="💬",
    layout="centered",
)

st.title("Document Q&A Chatbot")
st.caption("Datheon Python + Generative AI Internship · Task 4")
st.write(
    "Ask questions about the indexed sample document. "
    "Answers in live mode use retrieved evidence and show their sources."
)


@st.cache_resource
def get_retriever():
    """Load and index the sample document once per app session."""
    document_path = (
        ROOT / "task2_rag_assistant" / "documents" / "sample_document.txt"
    )
    if not document_path.exists():
        raise FileNotFoundError(
            f"Sample document not found: {document_path}"
        )

    pages = load_document(document_path)
    chunks = chunk_pages(pages, chunk_size=400, chunk_overlap=60)

    retriever = LocalRetriever(
        persist_directory=ROOT / "task4_streamlit_chatbot" / "storage",
        collection_name="task4_sample_document",
    )
    retriever.add_chunks(chunks)
    return retriever


def demo_stream(passages):
    """Stream retrieved evidence without making an LLM API request."""
    if not passages:
        yield "I could not find relevant evidence in the indexed document."
        return

    text = (
        "Demo mode: this is retrieved evidence, not an AI-generated answer.\n\n"
        + passages[0]["text"]
    )
    words = text.split()
    for index in range(0, len(words), 8):
        yield " ".join(words[index:index + 8]) + " "


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "Response mode",
        ["Demo (no API calls)", "Live Gemini"],
        help=(
            "Demo streams retrieved text without an LLM. "
            "Live Gemini generates a grounded answer and may use API quota."
        ),
    )
    st.caption("The sample document is indexed locally.")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(f"- {source}")

try:
    retriever = get_retriever()
    st.caption("Sample document ready.")
except Exception as exc:
    st.error(f"Could not prepare the document index: {exc}")
    st.stop()

question = st.chat_input("Ask a question about the sample document...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )
    with st.chat_message("user"):
        st.markdown(question)

    try:
        passages = retriever.search(question, top_k=3)
        sources = []
        seen = set()

        for passage in passages:
            metadata = passage.get("metadata", {})
            source = metadata.get("source", "Unknown source")
            page = metadata.get("page_number")
            label = f"{source}, page {page}" if page else source
            if label not in seen:
                seen.add(label)
                sources.append(label)

        with st.chat_message("assistant"):
            if mode == "Demo (no API calls)":
                answer = st.write_stream(demo_stream(passages))
            elif not os.getenv("GEMINI_API_KEY"):
                answer = (
                    "Live mode requires GEMINI_API_KEY. "
                    "Add your key to the local .env file, or select Demo mode."
                )
                st.markdown(answer)
            elif not passages:
                answer = (
                    "I could not find relevant evidence in the indexed document."
                )
                st.markdown(answer)
            else:
                prompt = build_answer_prompt(question, passages)
                answer = st.write_stream(stream_gemini_answer(prompt))

            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.markdown(f"- {source}")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )
    except Exception as exc:
        st.error(
            "The response could not be completed. "
            f"Error type: {type(exc).__name__}. "
            "Check your configuration and try again."
        )
