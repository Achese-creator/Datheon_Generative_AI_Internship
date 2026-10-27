"""Autonomous, bounded tool-calling agent for Task 3."""

import os
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from task3_tool_agent.tools import TASK3_TOOLS

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

SYSTEM_PROMPT = """You are a research assistant that solves multi-step questions
using tools when appropriate.

Available tools:
- search_web: find current web information and source URLs.
- calculate: perform mathematical calculations.
- read_local_file: read a named TXT or Markdown file in your knowledge folder.

Instructions:
1. Decide which tools are needed based on the user's question.
2. Use tool results as evidence; do not invent search results or calculations.
3. For compound questions, gather the necessary evidence before answering.
4. Cite web source URLs when web search results support your answer.
5. If a tool fails or evidence is insufficient, explain the limitation.
6. Clearly distinguish verified facts, estimates, and assumptions.
7. Provide a concise, useful final answer.
8. Never claim to have used a tool unless it was actually called.
"""

MAX_TOOL_ROUNDS = 3
MAX_DISPLAY_CHARS = 1_500


def create_model():
    """Create the Gemini chat model using the project's existing settings."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to the project .env file."
        )

    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        max_retries=1,
        timeout=60,
    )


def _content_to_text(content) -> str:
    """Convert common model response formats into printable text."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)

    return str(content)


def _display_observation(content: str) -> str:
    """Keep console output readable without truncating evidence sent to the model."""
    if len(content) <= MAX_DISPLAY_CHARS:
        return content
    return content[:MAX_DISPLAY_CHARS] + "\n[Console preview truncated; full result retained for the agent.]"


def run_agent(
    query: str,
    model=None,
    max_tool_rounds: int = MAX_TOOL_ROUNDS,
    emit: Callable[[str], None] = print,
) -> str:
    """Run a bounded autonomous tool-calling loop.

    The model chooses tools. Each result is returned to the model as an
    observation before it decides whether more tools or a final answer are needed.
    """
    if not query.strip():
        raise ValueError("The query cannot be empty.")

    if max_tool_rounds < 0:
        raise ValueError("max_tool_rounds cannot be negative.")

    base_model = model if model is not None else create_model()
    tool_enabled_model = base_model.bind_tools(TASK3_TOOLS)
    tools_by_name = {tool.name: tool for tool in TASK3_TOOLS}

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query.strip()),
    ]

    tool_rounds = 0

    emit("\n[THOUGHT] The agent is assessing the question and available tools.")
    emit("[Note] Internal chain-of-thought is not displayed; tool decisions and observations are logged.")

    while True:
        response = tool_enabled_model.invoke(messages)

        tool_calls = getattr(response, "tool_calls", None) or []

        if not tool_calls:
            answer = _content_to_text(response.content).strip()
            emit("\n[FINAL ANSWER]")
            emit(answer or "The model returned an empty response.")
            return answer

        if tool_rounds >= max_tool_rounds:
            emit(
                f"\n[LIMIT] Reached the maximum of {max_tool_rounds} tool rounds. "
                "Stopping to control runtime and API usage."
            )
            return (
                "The agent reached its tool-round limit before producing a final "
                "answer. Try a narrower question or increase the limit."
            )

        # Preserve the assistant tool-call message for the model's conversation.
        messages.append(response)

        for call in tool_calls:
            tool_name = call.get("name", "")
            tool_args = call.get("args", {})
            tool_call_id = call.get("id", "")

            emit(f"\n[ACTION] {tool_name}")
            emit(f"Arguments: {tool_args}")

            selected_tool = tools_by_name.get(tool_name)

            if selected_tool is None:
                observation = f"Tool error: unknown tool '{tool_name}'."
            else:
                try:
                    observation = str(selected_tool.invoke(tool_args))
                except Exception as exc:
                    observation = (
                        f"Tool error: {type(exc).__name__}: {exc}"
                    )

            emit("[OBSERVATION]")
            emit(_display_observation(observation))

            messages.append(
                ToolMessage(
                    content=observation,
                    tool_call_id=tool_call_id,
                )
            )

        tool_rounds += 1


def main():
    """Provide a simple interactive console."""
    print("Task 3 — Multi-Tool Autonomous AI Agent")
    print("Tools: DuckDuckGo search, calculator, local file reader")
    print("Enter a question, or type 'exit' to quit.")

    try:
        model = create_model()
    except Exception as exc:
        print(f"\nStartup error: {exc}")
        return

    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if query.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        if not query:
            continue

        try:
            run_agent(query, model=model)
        except Exception as exc:
            print(f"\nAgent error: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()


