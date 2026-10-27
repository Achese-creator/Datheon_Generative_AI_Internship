from task3_tool_agent.tools import calculate, read_local_file

def test_calculator_addition():
    assert "2 + 3 = 5" in calculate.invoke({"expression": "2 + 3"})

def test_calculator_math_function():
    assert "sqrt(81) = 9.0" in calculate.invoke({"expression": "sqrt(81)"})

def test_calculator_rejects_python_execution():
    result = calculate.invoke({
        "expression": "__import__('os').system('whoami')"
    })
    assert "error" in result.lower()

def test_calculator_division_by_zero():
    result = calculate.invoke({"expression": "1 / 0"})
    assert "error" in result.lower()

def test_reader_reads_knowledge_file():
    result = read_local_file.invoke({"filename": "agent_notes.md"})
    assert "three tools" in result.lower()

def test_reader_rejects_path_traversal():
    result = read_local_file.invoke({"filename": "..\\..\\.env"})
    assert "denied" in result.lower()

def test_reader_rejects_unsupported_extension():
    result = read_local_file.invoke({"filename": "secrets.env"})
    assert "denied" in result.lower()
