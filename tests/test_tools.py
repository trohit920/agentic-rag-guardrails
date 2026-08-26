"""Unit tests for the calculator tool and query routing helpers (pure logic)."""

from __future__ import annotations

import math

import pytest

from src.tools import CalculatorError, extract_expression, tool_calculator


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("4*60", 240.0),
        ("2 + 3 * 4", 14.0),
        ("(1 + 2) * 3", 9.0),
        ("-5 + 2", -3.0),
        ("10 / 4", 2.5),
    ],
)
def test_calculator_evaluates_arithmetic(expression: str, expected: float) -> None:
    """The calculator evaluates whitelisted arithmetic correctly."""
    assert math.isclose(tool_calculator(expression), expected)


@pytest.mark.parametrize("bad", ["__import__('os')", "1 +", "a + b", "open('x')"])
def test_calculator_rejects_unsafe_input(bad: str) -> None:
    """Non-arithmetic or malformed input raises CalculatorError, never executes."""
    with pytest.raises(CalculatorError):
        tool_calculator(bad)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("what is 4 * 60?", "4 * 60"),
        ("calculate 2+2", "2+2"),
        ("12*5", "12*5"),
    ],
)
def test_extract_expression_detects_calculations(query: str, expected: str) -> None:
    """Arithmetic-style queries yield their expression for calculator routing."""
    assert extract_expression(query) == expected


@pytest.mark.parametrize(
    "query",
    [
        "What is the meal allowance?",
        "A manager takes a four-day domestic trip.",
        "How long are records retained?",
    ],
)
def test_extract_expression_ignores_natural_language(query: str) -> None:
    """Natural-language questions are not routed to the calculator."""
    assert extract_expression(query) is None
