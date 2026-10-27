from langchain_core.messages import AIMessage

from task3_tool_agent.agent import run_agent


class FakeModel:
    """Deterministic model substitute for tests without API calls."""

    def __init__(self, responses):
        self.responses = iter(responses)

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        return next(self.responses)


def test_agent_invokes_calculator_and_returns_final_answer():
    model = FakeModel([
        AIMessage(
            content="",
            tool_calls=[{
                "name": "calculate",
                "args": {"expression": "2 + 3"},
                "id": "call-1",
                "type": "tool_call",
            }],
        ),
        AIMessage(content="The answer is 5."),
    ])
    logs = []

    answer = run_agent(
        "What is 2 + 3?",
        model=model,
        emit=logs.append,
    )

    assert answer == "The answer is 5."
    assert any("[ACTION] calculate" in line for line in logs)
    assert any("2 + 3 = 5" in line for line in logs)
    assert any("[FINAL ANSWER]" in line for line in logs)


def test_agent_stops_when_tool_round_limit_is_reached():
    model = FakeModel([
        AIMessage(
            content="",
            tool_calls=[{
                "name": "calculate",
                "args": {"expression": "2 + 3"},
                "id": "call-1",
                "type": "tool_call",
            }],
        ),
        AIMessage(
            content="",
            tool_calls=[{
                "name": "calculate",
                "args": {"expression": "4 + 5"},
                "id": "call-2",
                "type": "tool_call",
            }],
        ),
    ])
    logs = []

    answer = run_agent(
        "Calculate two things.",
        model=model,
        max_tool_rounds=1,
        emit=logs.append,
    )

    assert "tool-round limit" in answer
    assert sum("[ACTION] calculate" in line for line in logs) == 1
    assert any("[LIMIT]" in line for line in logs)
