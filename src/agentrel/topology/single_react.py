"""Single ReAct agent — one LLM loop that reasons, calls a tool, reads the
result, and repeats until it answers. Matches ChatGPT / Claude single-agent
data analysis (report Table 1). Structurally a linear chain, which the
robustness literature predicts is the most fragile under faults."""
from __future__ import annotations

from ..agents.tools import Session
from ..harness.trace import RunTrace
from .base import ACTION_PROTOCOL, Topology, initial_history


class SingleReAct(Topology):
    name = "single_react"

    def run(self, trace: RunTrace, session: Session) -> None:
        system = ACTION_PROTOCOL
        history = initial_history(trace, session)
        step = 0
        while step < self.max_steps:
            action = self._ask(trace, system, history)
            if action.get("action") == "final":
                trace.finish(str(action.get("answer", "")))
                return
            tool = action.get("tool", "")
            args = action.get("args", {}) or {}
            ok = self._exec_tool(trace, session, role="react", step=step,
                                 tool=tool, args=args)
            last = trace.calls[-1]
            history += (f"\nStep {step}: {tool}({args}) -> "
                        f"{'OK ' + last.output_repr if ok else 'ERROR ' + str(last.error)}")
            # ReAct's only recovery: retry once on a loud error
            if not ok:
                ok2 = self._exec_tool(trace, session, role="react", step=step,
                                      tool=tool, args=args, is_retry=True)
                history += f"\n  retry -> {'OK' if ok2 else 'ERROR again'}"
            step += 1
        # ran out of steps without answering
        trace.timed_out = True
        trace.finish(str(session.scratch.get("last_value", "")))
