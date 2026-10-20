"""Orchestrator-worker topology — a planner issues sub-steps to a worker and an
inspector reviews the result before it is accepted. Matches LAMBDA's
programmer/inspector split (report Table 1). Hierarchical, with a verification
node, which the robustness literature predicts degrades more gracefully than a
linear chain."""
from __future__ import annotations

from ..agents.tools import Session
from ..harness.trace import RunTrace
from .base import ACTION_PROTOCOL, Topology

ORCH_SYSTEM = (ACTION_PROTOCOL + "\nYou are the ORCHESTRATOR. Decide the next tool "
               "call that moves toward the answer, or finalise.")
INSPECT_SYSTEM = (ACTION_PROTOCOL + "\nYou are the INSPECTOR. Given the last tool "
                  "result, respond {\"action\":\"accept\"} or "
                  "{\"action\":\"redo\"} if it looks wrong.")


class OrchestratorWorker(Topology):
    name = "orchestrator_worker"

    def run(self, trace: RunTrace, session: Session) -> None:
        history = f"Question: {trace.question_text}\n"
        step = 0
        while step < self.max_steps:
            action = self._ask(trace, ORCH_SYSTEM, history)
            if action.get("action") == "final":
                trace.finish(str(action.get("answer", "")))
                return
            tool = action.get("tool", "")
            args = action.get("args", {}) or {}

            # worker executes
            ok = self._exec_tool(trace, session, role="worker", step=step,
                                 tool=tool, args=args)
            last = trace.calls[-1]
            history += (f"\nStep {step} worker: {tool}({args}) -> "
                        f"{'OK ' + last.output_repr if ok else 'ERROR ' + str(last.error)}")

            # inspector reviews (its own LLM turn — a verification node)
            verdict = self._ask(trace, INSPECT_SYSTEM, history + "\nInspect the above.")
            if verdict.get("action") == "redo" or not ok:
                self._exec_tool(trace, session, role="worker", step=step,
                                tool=tool, args=args, is_retry=True)
                history += "\n  inspector: redo"
            else:
                history += "\n  inspector: accept"
            step += 1
        trace.timed_out = True
        trace.finish(str(session.scratch.get("last_value", "")))
