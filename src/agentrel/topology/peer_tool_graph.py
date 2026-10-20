"""Peer tool-graph topology — several specialist peers (a loader, a transformer,
a reducer) hand work between each other based on the state of the analysis,
rather than through a central controller. Closest to Data Interpreter's task
graph (report Table 1). A closed-loop / cyclic graph, which the robustness
literature associates with higher fault-neutralisation."""
from __future__ import annotations

from ..agents.tools import Session
from ..harness.trace import RunTrace
from .base import ACTION_PROTOCOL, Topology, initial_history

# Each peer owns a subset of tools and routes to the next peer by analysis stage.
PEERS = {
    "loader": {"tools": ["list_tables", "load_table", "join_tables"], "next": "transformer"},
    "transformer": {"tools": ["filter_rows", "compute_missing_rate"], "next": "reducer"},
    "reducer": {"tools": ["aggregate", "group_aggregate"], "next": "loader"},
}


def _peer_system(peer: str) -> str:
    allowed = ", ".join(PEERS[peer]["tools"])
    return (ACTION_PROTOCOL + f"\nYou are the {peer.upper()} peer. You may call: "
            f"{allowed}. If your part is done, respond "
            '{"action":"handoff"}. If the whole question is answered, finalise.')


class PeerToolGraph(Topology):
    name = "peer_tool_graph"

    def run(self, trace: RunTrace, session: Session) -> None:
        history = initial_history(trace, session)
        peer = "loader"
        step = 0
        handoffs = 0
        while step < self.max_steps and handoffs < self.max_steps:
            action = self._ask(trace, _peer_system(peer), history)
            act = action.get("action")
            if act == "final":
                trace.finish(str(action.get("answer", "")))
                return
            if act == "handoff":
                peer = PEERS[peer]["next"]
                handoffs += 1
                history += f"\n-> handoff to {peer}"
                continue
            tool = action.get("tool", "")
            args = action.get("args", {}) or {}
            # a peer calling a tool outside its set is itself a (silent) routing error
            ok = self._exec_tool(trace, session, role=peer, step=step,
                                 tool=tool, args=args)
            last = trace.calls[-1]
            history += (f"\nStep {step} {peer}: {tool}({args}) -> "
                        f"{'OK ' + last.output_repr if ok else 'ERROR ' + str(last.error)}")
            if not ok:
                self._exec_tool(trace, session, role=peer, step=step,
                                tool=tool, args=args, is_retry=True)
            step += 1
        trace.timed_out = True
        trace.finish(str(session.scratch.get("last_value", "")))


TOPOLOGIES = {}  # populated in topology/__init__.py
