"""Tools available to the Task 3 autonomous agent."""

import ast
import math
from pathlib import Path

from ddgs import DDGS
from langchain.tools import tool

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"
MAX_FILE_CHARS = 12_000
MAX_EXPRESSION_LENGTH = 200

_ALLOWED_BINARY_OPERATORS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}
_ALLOWED_UNARY_OPERATORS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}
_ALLOWED_FUNCTIONS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "ceil": math.ceil,
    "floor": math.floor,
}
_ALLOWED_CONSTANTS = {"pi": math.pi, "e": math.e}


def _evaluate_node(node):
    """Evaluate only explicitly permitted arithmetic AST nodes."""
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body)

    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value

    if isinstance(node, ast.Name) and node.id in _ALLOWED_CONSTANTS:
        return _ALLOWED_CONSTANTS[node.id]

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINARY_OPERATORS:
        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)

        if isinstance(node.op, ast.Pow):
            if abs(right) > 100:
                raise ValueError("Exponent magnitude must not exceed 100.")

        result = _ALLOWED_BINARY_OPERATORS[type(node.op)](left, right)
        if isinstance(result, complex) or not math.isfinite(result):
            raise ValueError("The result must be a finite real number.")
        return result

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPERATORS:
        return _ALLOWED_UNARY_OPERATORS[type(node.op)](
            _evaluate_node(node.operand)
        )

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        function = _ALLOWED_FUNCTIONS.get(node.func.id)
        if function is None or node.keywords or len(node.args) != 1:
            raise ValueError("Unsupported function or arguments.")
        return function(_evaluate_node(node.args[0]))

    raise ValueError("Unsupported expression. Use basic arithmetic or an allowed math function.")


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression. Supports arithmetic, pi, e, sqrt, log, sin, cos, tan, ceil and floor."""
    if len(expression) > MAX_EXPRESSION_LENGTH:
        return "Calculation error: expression is too long."

    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate_node(tree)
        return f"{expression} = {result}"
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
        return f"Calculation error: {exc}"


@tool
def search_web(query: str) -> str:
    """Search the live web using DuckDuckGo. Use for current facts and research; include source titles and URLs."""
    try:
        with DDGS() as search:
            results = list(search.text(query, max_results=5))

        if not results:
            return "No web results were found. Try a more specific query."

        formatted = []
        for index, result in enumerate(results, start=1):
            title = result.get("title", "Untitled")
            url = result.get("href", "")
            snippet = result.get("body", "")
            formatted.append(
                f"{index}. {title}\nURL: {url}\nSummary: {snippet}"
            )
        return "\n\n".join(formatted)
    except Exception as exc:
        return f"Web search failed: {type(exc).__name__}: {exc}"


@tool
def read_local_file(filename: str) -> str:
    """Read a TXT or Markdown file from the agent's knowledge folder. Supply only the filename, not an arbitrary path."""
    if Path(filename).name != filename or filename in {".", ".."}:
        return "File read denied: provide a filename located directly in the knowledge folder."

    path = (KNOWLEDGE_DIR / filename).resolve()

    if path.parent != KNOWLEDGE_DIR.resolve():
        return "File read denied: path is outside the knowledge folder."

    if path.suffix.lower() not in {".txt", ".md"}:
        return "File read denied: only .txt and .md files are supported."

    if not path.is_file():
        return f"File not found in knowledge folder: {filename}"

    try:
        content = path.read_text(encoding="utf-8")
        if len(content) > MAX_FILE_CHARS:
            content = content[:MAX_FILE_CHARS] + "\n\n[Truncated at the file-reading limit.]"
        return f"Source file: {filename}\n\n{content}"
    except (OSError, UnicodeError) as exc:
        return f"File read failed: {type(exc).__name__}: {exc}"


TASK3_TOOLS = [search_web, calculate, read_local_file]
