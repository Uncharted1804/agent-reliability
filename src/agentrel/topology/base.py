"""Shared machinery for the three agent topologies.

A topology is a policy for turning a question + a set of tools into a sequence
of tool calls and a final answer, while recording a RunTrace. All three share:
  * the same base LLM (fixed, temp 0)
  * the same tool registry
  * the same JSON action protocol the model is asked to follow
They differ only in HOW control flows between agent roles — which is exactly the
variable the robustness study isolates.
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod

from ..agents.llm import LLM
from ..agents.tools import Session, call_tool
from ..harness.trace import RunTrace, ToolCall

ACTION_PROTOCOL = """You are a data-analysis agent. Respond with ONE JSON object.
To call a tool: {"action":"tool","tool":"<name>","args":{...}}
To answer:      {"action":"final","answer":"<value>"}
Available tools: list_tables, load_table, filter_rows, aggregate,
group_aggregate, join_tables, compute_missing_rate.
Answer with only the value (a number or a single category label)."""


def parse_action(text: str) -> dict:
    """Best-effort parse of the model's JSON action. Tolerates stray prose."""
    text = (text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {"action": "final", "answer": text}


class Topology(ABC):
    name: str = "base"

    def __init__(self, llm: LLM, max_steps: int = 12):
        self.llm = llm
        self.max_steps = max_steps

    @abstractmethod
    def run(self, trace: RunTrace, session: Session) -> None:
        """Drive the agent to completion, mutating `trace` in place."""

    # -- helpers shared by subclasses -------------------------------------- #
    def _ask(self, trace: RunTrace, system: str, user: str) -> dict:
        resp = self.llm.complete(system, user)
        trace.n_llm_calls += 1
        trace.prompt_tokens += resp.prompt_tokens
        trace.completion_tokens += resp.completion_tokens
        return parse_action(resp.text)

    def _exec_tool(self, trace: RunTrace, session: Session, *, role: str,
                   step: int, tool: str, args: dict, is_retry: bool = False) -> bool:
        t0 = time.time()
        result = call_tool(tool, session, args)
        trace.add_call(ToolCall(
            step=step, agent_role=role, tool=tool, args=args,
            output_repr=result.repr if result.ok else "",
            ok=result.ok, error=result.error, is_retry=is_retry,
            latency_s=round(time.time() - t0, 4),
        ))
        return result.ok
