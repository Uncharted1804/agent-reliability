"""Fault injection (block E) — breaking one component of a running agent.

The robustness study asks how much each topology degrades when a single
component misbehaves. Three faults, matching the `faults:` block of
experiment.yaml:

    tool_failure      a tool call returns an error instead of its result.
                      The *loud* fault: the agent can see it and may recover.
    corrupted_output  a tool's result is silently mutated before the agent sees
                      it. The dangerous fault: nothing looks wrong, so the agent
                      has no reason to retry and carries the bad value forward.
    dropped_context   the prior step's result is withheld from the model's
                      prompt. The agent must cope with amnesia rather than with
                      an error.

Design choices that matter for the estimate:

* One fault per run, at one step, so Delta_{T,f} attributes degradation to a
  single break rather than to a storm of them.
* The target step is drawn from a seed derived from the run id and the fault
  name, so the same run is faulted identically on every re-execution — the
  corpus is reproducible, and baseline/faulted pairs are comparable.
* A run can finish before reaching its target step. Such a run is *not* a
  faulted observation, so `realised` is tracked and the robustness layer keeps
  only realised runs. Silently treating unrealised runs as faulted would bias
  Delta_{T,f} toward zero.
* Corruption writes back into the session scratch, so a corrupted intermediate
  value propagates exactly as a real one would.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..agents.tools import Session, ToolResult

FAULTS = ("tool_failure", "corrupted_output", "dropped_context")

# The target step is drawn from [0, TARGET_WINDOW). Most runs make only a
# handful of tool calls, so a wider window would leave most faults unrealised.
TARGET_WINDOW = 3

# Multiplicative perturbation for corrupted numeric outputs. Large enough to
# break the 1% labelling tolerance, small enough to stay plausible — a
# corruption the agent would obviously notice is not a *silent* fault.
CORRUPTION_FACTOR = 1.35

_REDACTED = "[result unavailable]"


def _seeded_target(run_id: str, fault: str, window: int = TARGET_WINDOW) -> int:
    """Deterministic target step for (run, fault)."""
    digest = hashlib.sha256(f"{run_id}|{fault}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % window


def corrupt_value(value):
    """Silently mutate a tool's value, keeping its type and plausibility."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        corrupted = value * CORRUPTION_FACTOR
        return int(round(corrupted)) if isinstance(value, int) else corrupted
    if isinstance(value, dict):
        # tool results are small dicts; perturb the numeric entries
        out = dict(value)
        for key, val in value.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                out[key] = corrupt_value(val)
        return out
    if isinstance(value, str):
        return value
    return value


@dataclass
class FaultInjector:
    """Applies one fault to one run. Stateful: fires at most once."""

    fault: str | None
    run_id: str
    window: int = TARGET_WINDOW

    def __post_init__(self) -> None:
        if self.fault is not None and self.fault not in FAULTS:
            raise ValueError(f"unknown fault {self.fault!r}; known: {FAULTS}")
        self.target_step: int | None = (
            None if self.fault is None else _seeded_target(self.run_id, self.fault,
                                                           self.window))
        self.realised: bool = False

    # -- seams the topologies call ----------------------------------------- #
    def on_tool_result(self, step: int, tool: str, result: ToolResult,
                       session: Session) -> ToolResult:
        """Rewrite a tool result if this step is the target of a result fault."""
        if self.fault is None or self.realised or step != self.target_step:
            return result
        if self.fault == "tool_failure":
            self.realised = True
            return ToolResult(ok=False, value=None,
                              error="InjectedToolFailure: tool unavailable", repr="")
        if self.fault == "corrupted_output":
            if not result.ok:
                return result       # nothing meaningful to corrupt
            self.realised = True
            corrupted = corrupt_value(result.value)
            # propagate exactly as a genuine value would
            if "last_value" in session.scratch and not isinstance(corrupted, dict):
                session.scratch["last_value"] = corrupted
            return ToolResult(ok=True, value=corrupted, error=None,
                              repr=repr(corrupted)[:200])
        return result

    def on_prompt(self, user: str) -> str:
        """Withhold the most recent step's result from the model's prompt."""
        if self.fault != "dropped_context" or self.realised:
            return user
        marker = user.rfind(" -> ")
        if marker == -1:
            return user             # no step result in the prompt yet
        end = user.find("\n", marker)
        end = len(user) if end == -1 else end
        # the target step indexes which observed result gets dropped
        if user.count(" -> ") <= self.target_step:
            return user
        self.realised = True
        return user[:marker] + " -> " + _REDACTED + user[end:]

    def describe(self) -> dict[str, object]:
        """What to record on the trace."""
        return {"injected_fault": self.fault, "fault_target_step": self.target_step,
                "fault_realised": self.realised}
