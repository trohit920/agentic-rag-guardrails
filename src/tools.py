"""Agent tools the router can select between.

Each tool is a small callable with a clear signature so the LangGraph router can
choose the right one for a query: general retrieval, department-scoped retrieval,
or a safe arithmetic calculator. Demonstrates the "agent that chooses between
multiple tools/APIs" requirement with no external services. The calculator uses an
AST evaluator (never Python ``eval``) so untrusted input cannot execute code.
"""

from __future__ import annotations

import ast
import operator
import re
from collections.abc import Callable

from .retriever import Retriever
from .schemas import RetrievalResult, Sensitivity

_BINARY_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_EXPRESSION_RE = re.compile(r"[\d.]+(?:\s*[-+*/%()]\s*[\d.]+)+")
_LEADING_PHRASE_RE = re.compile(r"^(what\s+is|calculate|compute|how\s+much\s+is)\s+", re.IGNORECASE)


class CalculatorError(ValueError):
    """Raised when an expression is not a safe, well-formed arithmetic formula."""


def tool_retrieve(
    retriever: Retriever, query: str, max_sensitivity: Sensitivity
) -> list[RetrievalResult]:
    """General document retrieval tool."""
    return retriever.retrieve(query, max_sensitivity)


def tool_metadata_retrieve(
    retriever: Retriever, query: str, max_sensitivity: Sensitivity, department: str
) -> list[RetrievalResult]:
    """Retrieval scoped to a specific department via metadata filtering."""
    return retriever.retrieve(query, max_sensitivity, department=department)


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate a whitelisted arithmetic AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        return _BINARY_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise CalculatorError("expression contains an unsupported operation")


def tool_calculator(expression: str) -> float:
    """Safely evaluate an arithmetic expression (AST-based, never ``eval``)."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"invalid expression: {expression!r}") from exc
    return _eval_node(tree.body)


def extract_expression(query: str) -> str | None:
    """Return the arithmetic expression in a query if it is a calculation, else None."""
    stripped = _LEADING_PHRASE_RE.sub("", query.strip()).rstrip("?= ").strip()
    match = _EXPRESSION_RE.fullmatch(stripped)
    return stripped if match else None
