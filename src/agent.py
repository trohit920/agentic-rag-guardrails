"""LangGraph agent orchestrating the end-to-end RAG flow.

Wires the pipeline as a graph: input injection-guard -> tool router -> retrieve
(ACL-filtered) or calculate -> token-budgeted context assembly -> local-LLM
generation -> output groundedness/abstention guard. Conditional edges handle
adversarial input (safe refusal), calculator routing, and ungrounded answers
(abstain). Guards are optional dependencies so the retrieval+generation core can
run and be tested before the guardrail phase adds them.
"""

from __future__ import annotations

import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .config import ABSTAIN_MESSAGE, Sensitivity
from .context import assemble_context
from .guardrails import GroundednessGuard, InjectionGuard
from .llm import LocalLLM
from .retriever import Retriever
from .schemas import AgentAnswer, GuardVerdict, RetrievalResult
from .tools import extract_expression, tool_calculator, tool_retrieve

_CITATION_RE = re.compile(r"\[(\d+)\]")
_ROUTE_RETRIEVE = "retrieve"
_ROUTE_CALCULATE = "calculate"
_ROUTE_REFUSE = "refuse"


class AgentState(TypedDict, total=False):
    """Mutable state threaded through the LangGraph nodes."""

    query: str
    user_clearance: Sensitivity
    history: list[dict[str, str]]
    route: str
    retrieved: list[RetrievalResult]
    messages: list[dict[str, str]]
    answer: str
    citations: list[str]
    abstained: bool
    verdicts: list[GuardVerdict]
    latency_seconds: float
    completion_tokens: int


class RagAgent:
    """Compiled LangGraph agent exposing a single answer() entry point."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LocalLLM,
        injection_guard: InjectionGuard | None = None,
        groundedness_guard: GroundednessGuard | None = None,
    ) -> None:
        """Build and compile the LangGraph graph from the supplied components."""
        self.retriever = retriever
        self.llm = llm
        self.injection_guard = injection_guard
        self.groundedness_guard = groundedness_guard
        self._graph = self._build_graph()

    def _build_graph(self):
        """Assemble and compile the state graph."""
        graph = StateGraph(AgentState)
        graph.add_node("guard_input", self._guard_input)
        graph.add_node("route", self._route)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("assemble", self._assemble)
        graph.add_node("generate", self._generate)
        graph.add_node("guard_output", self._guard_output)
        graph.add_node("calculate", self._calculate)

        graph.add_edge(START, "guard_input")
        graph.add_conditional_edges(
            "guard_input", self._after_input_guard, {"continue": "route", _ROUTE_REFUSE: END}
        )
        graph.add_conditional_edges(
            "route",
            lambda state: state["route"],
            {_ROUTE_RETRIEVE: "retrieve", _ROUTE_CALCULATE: "calculate"},
        )
        graph.add_edge("retrieve", "assemble")
        graph.add_edge("assemble", "generate")
        graph.add_edge("generate", "guard_output")
        graph.add_edge("guard_output", END)
        graph.add_edge("calculate", END)
        return graph.compile()

    def _guard_input(self, state: AgentState) -> dict:
        """Reject prompt-injection / jailbreak attempts before answering."""
        if self.injection_guard is None:
            return {}
        verdict = self.injection_guard.check(state["query"])
        if verdict.passed:
            return {"verdicts": [verdict]}
        return {
            "route": _ROUTE_REFUSE,
            "abstained": True,
            "answer": ABSTAIN_MESSAGE,
            "citations": [],
            "verdicts": [verdict],
        }

    @staticmethod
    def _after_input_guard(state: AgentState) -> str:
        """Branch to a safe refusal when the input guard flagged the query."""
        return _ROUTE_REFUSE if state.get("route") == _ROUTE_REFUSE else "continue"

    @staticmethod
    def _route(state: AgentState) -> dict:
        """Pick a tool: calculator for pure arithmetic, else document retrieval."""
        if extract_expression(state["query"]) is not None:
            return {"route": _ROUTE_CALCULATE}
        return {"route": _ROUTE_RETRIEVE}

    def _calculate(self, state: AgentState) -> dict:
        """Evaluate an arithmetic query with the calculator tool."""
        expression = extract_expression(state["query"])
        result = tool_calculator(expression) if expression else float("nan")
        text = str(int(result)) if result == int(result) else str(result)
        return {"answer": text, "citations": [], "abstained": False}

    def _retrieve(self, state: AgentState) -> dict:
        """Retrieve ACL-filtered context for the query."""
        results = tool_retrieve(self.retriever, state["query"], state["user_clearance"])
        return {"retrieved": results}

    def _assemble(self, state: AgentState) -> dict:
        """Pack retrieved context and history into a token-bounded prompt."""
        messages = assemble_context(
            state["query"],
            state.get("retrieved", []),
            state.get("history", []),
            self.llm.count_tokens,
        )
        return {"messages": messages}

    def _generate(self, state: AgentState) -> dict:
        """Generate a grounded answer and map its [n] citations to chunk ids."""
        generation = self.llm.generate(state["messages"])
        retrieved = state.get("retrieved", [])
        cited: list[str] = []
        for number in _CITATION_RE.findall(generation.text):
            index = int(number) - 1
            if 0 <= index < len(retrieved):
                cited.append(retrieved[index].chunk.chunk_id)
        return {
            "answer": generation.text,
            "citations": list(dict.fromkeys(cited)),
            "latency_seconds": generation.latency_seconds,
            "completion_tokens": generation.completion_tokens,
        }

    def _guard_output(self, state: AgentState) -> dict:
        """Abstain when the answer is not grounded in the retrieved context."""
        if self.groundedness_guard is None:
            return {}
        report = self.groundedness_guard.score(state["answer"], state.get("retrieved", []))
        verdict = self.groundedness_guard.verdict(report)
        verdicts = [*state.get("verdicts", []), verdict]
        if verdict.passed:
            return {"verdicts": verdicts}
        return {"answer": ABSTAIN_MESSAGE, "citations": [], "abstained": True, "verdicts": verdicts}

    def answer(
        self,
        query: str,
        user_clearance: Sensitivity,
        history: list[dict[str, str]] | None = None,
    ) -> AgentAnswer:
        """Run a query through the full graph and return the guarded answer."""
        final: AgentState = self._graph.invoke(
            {"query": query, "user_clearance": user_clearance, "history": history or []}
        )
        return AgentAnswer(
            query=query,
            answer=final.get("answer", ""),
            citations=final.get("citations", []),
            abstained=final.get("abstained", False),
            verdicts=final.get("verdicts", []),
            latency_seconds=final.get("latency_seconds", 0.0),
            completion_tokens=final.get("completion_tokens", 0),
        )
