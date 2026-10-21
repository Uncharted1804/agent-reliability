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
from ..agents.tools import Session, available_tools, call_tool, schema_digest
from ..harness.trace import RunTrace, ToolCall

# The protocol must document each tool's ARGUMENTS, not just its name. With
# names alone the model cannot form a valid call, so it skips tools and guesses
# the answer outright — which yields an empty tool-interaction graph and makes
# the graph features (and therefore the classifier) degenerate.
ACTION_PROTOCOL = """You are a data-analysis agent working on a real table.
Respond with ONE JSON object and nothing else.

To call a tool: {"action":"tool","tool":"<name>","args":{...}}
To answer:      {"action":"final","answer":"<value>"}

Tools and their exact arguments:
  list_tables          {}  -> row counts and columns of every table
  load_table           {"table":"<name>"}  -> make that table the working frame
  filter_rows          {"query":"<pandas query>"}  e.g. "Sex == 'male' and Age > 30"
  aggregate            {"op":"mean|sum|median|count|max|min|nunique","column":"<col>"}
                       {"op":"quantile","column":"<col>","q":0.25}  q between 0 and 1
  group_aggregate      {"by":"<col>","op":"mean|sum|count|median|max|min",
                        "column":"<col>","pick":"idxmax|idxmin|value"}
  join_tables          {"left":"<name>","right":"<name>","on":"<col>","how":"left|inner"}
  compute_missing_rate {"column":"<col>"}  -> fraction of values missing

Rules:
  * Emit exactly ONE action per reply — the next single step, never a whole plan.
  * Compute the answer from the data using tools. Never guess a value from memory.
  * filter_rows and aggregate act on the working frame, so load_table first.
  * Column names are case-sensitive and must be taken from the schema you are given.
  * Once a tool has returned the value, answer with only that value: a bare
    number (rates as a fraction, e.g. 0.38) or a single category label."""


def initial_history(trace: RunTrace, session: Session) -> str:
    """Opening context shared by all three topologies.

    The schema is descriptive only (names, dtypes, cardinality, missing rate) —
    it never contains the answer, so every topology still has to do the
    analysis. Giving all three the same preamble keeps the topology the only
    variable the robustness study manipulates.
    """
    return (f"Schema of the data you may analyse:\n{schema_digest(session)}\n\n"
            f"Question: {trace.question_text}\n")


# Models deviate from the action protocol in a few predictable ways, and a
# parser that rejects them would record an empty trace — making the dynamic and
# graph features measure the parser's brittleness instead of the agent's
# behaviour. The deviations tolerated here are exactly the ones observed from a
# local model: a whole multi-step plan emitted at once (we take the first
# action and let the loop re-prompt), the tool name placed directly in
# "action", and tool arguments flattened to the top level instead of "args".
_ARG_KEYS = {"table", "query", "op", "column", "q", "by", "pick", "left", "right", "on", "how"}
_CONTROL_ACTIONS = {"accept", "redo", "handoff"}


def _first_json_object(text: str) -> dict | None:
    """First syntactically valid JSON object in the text, if any."""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def normalise_action(obj: dict) -> dict:
    """Map a loosely-shaped model action onto the protocol, or {} if it is not one."""
    act = str(obj.get("action", "")).strip().lower()
    tool = obj.get("tool")

    # the tool name used as the action verb: {"action":"load_table", ...}
    if not tool and act in available_tools():
        tool, act = act, "tool"

    if act == "tool" or tool:
        args = obj.get("args")
        if not isinstance(args, dict):
            # arguments flattened alongside the action
            args = {k: v for k, v in obj.items() if k in _ARG_KEYS}
        return {"action": "tool", "tool": str(tool or ""), "args": args}

    if act in ("final", "answer"):
        answer = obj.get("answer", obj.get("value", ""))
        return {"action": "final", "answer": answer}

    if act in _CONTROL_ACTIONS:
        return {"action": act}

    return {}


def parse_action(text: str) -> dict:
    """Best-effort parse of the model's JSON action. Tolerates stray prose.

    Falls back to treating the whole reply as a final answer, which keeps a
    malformed reply in the corpus as a (likely) failure rather than dropping
    the run — an unusable answer is itself the signal being measured.
    """
    text = (text or "").strip()
    obj = _first_json_object(text)
    if obj is not None:
        action = normalise_action(obj)
        if action:
            return action
    return {"action": "final", "answer": text}


class Topology(ABC):
    name: str = "base"

    def __init__(self, llm: LLM, max_steps: int = 12, injector=None):
        self.llm = llm
        self.max_steps = max_steps
        # optional FaultInjector (block E). None for baseline runs, so the
        # faulted and unfaulted corpora come from identical control flow.
        self.injector = injector

    @abstractmethod
    def run(self, trace: RunTrace, session: Session) -> None:
        """Drive the agent to completion, mutating `trace` in place."""

    # -- helpers shared by subclasses -------------------------------------- #
    def _ask(self, trace: RunTrace, system: str, user: str) -> dict:
        if self.injector is not None:
            user = self.injector.on_prompt(user)
        resp = self.llm.complete(system, user)
        trace.n_llm_calls += 1
        trace.prompt_tokens += resp.prompt_tokens
        trace.completion_tokens += resp.completion_tokens
        return parse_action(resp.text)

    def _exec_tool(self, trace: RunTrace, session: Session, *, role: str,
                   step: int, tool: str, args: dict, is_retry: bool = False) -> bool:
        t0 = time.time()
        result = call_tool(tool, session, args)
        injected = None
        if self.injector is not None:
            before = self.injector.realised
            result = self.injector.on_tool_result(step, tool, result, session)
            if self.injector.realised and not before:
                injected = self.injector.fault
        trace.add_call(ToolCall(
            step=step, agent_role=role, tool=tool, args=args,
            output_repr=result.repr if result.ok else "",
            ok=result.ok, error=result.error, is_retry=is_retry,
            latency_s=round(time.time() - t0, 4),
            injected_fault=injected,
        ))
        return result.ok
