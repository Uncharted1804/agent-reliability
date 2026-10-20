"""Execution-trace schema.

A trace is the ordered record of everything an agent did on one run: every tool
call with its arguments and output, every retry, every error, and the final
answer. This is the raw material the feature layer (static/dynamic/graph) reads
later, so the schema is deliberately explicit and JSON-serialisable.

One run == one (dataset, template, paraphrase, topology) combination.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A single tool invocation inside a run."""
    step: int                      # 0-based order within the run
    agent_role: str                # which agent/role issued it (topology-dependent)
    tool: str                      # tool name (node in the tool-interaction graph)
    args: dict[str, Any]
    output_repr: str = ""          # truncated string form of the output
    ok: bool = True                # did the tool call succeed
    error: str | None = None       # error class/message if not ok
    is_retry: bool = False         # a re-attempt of the previous failed call
    latency_s: float = 0.0
    injected_fault: str | None = None  # set by the fault-injection layer (later)


@dataclass
class RunTrace:
    run_id: str
    dataset: str
    template_id: str
    question_id: str
    paraphrase_index: int
    topology: str
    model: str
    question_text: str

    calls: list[ToolCall] = field(default_factory=list)
    final_answer_raw: str | None = None    # model's free-text final answer
    n_llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    timed_out: bool = False                # hit max_agent_steps / run_timeout
    harness_error: str | None = None       # the harness itself failed (not the agent)

    # fault injection (robustness study). A fault targets a step index chosen
    # before the run starts, so a run that ends early never reaches it —
    # `fault_realised` records whether the fault actually fired, and only
    # realised runs enter the Delta_{T,f} estimate.
    injected_fault: str | None = None
    fault_target_step: int | None = None
    fault_realised: bool = False

    def add_call(self, call: ToolCall) -> None:
        self.calls.append(call)

    def finish(self, final_answer_raw: str | None) -> None:
        self.final_answer_raw = final_answer_raw
        self.ended_at = time.time()

    @property
    def wall_time_s(self) -> float:
        return (self.ended_at or time.time()) - self.started_at

    @property
    def n_tool_calls(self) -> int:
        return len(self.calls)

    @property
    def n_retries(self) -> int:
        return sum(1 for c in self.calls if c.is_retry)

    @property
    def n_errors(self) -> int:
        return sum(1 for c in self.calls if not c.ok)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # derived convenience fields, so the raw corpus is self-describing
        d["n_tool_calls"] = self.n_tool_calls
        d["n_retries"] = self.n_retries
        d["n_errors"] = self.n_errors
        d["wall_time_s"] = round(self.wall_time_s, 4)
        return d
