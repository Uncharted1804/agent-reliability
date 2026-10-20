"""Topology registry."""
from __future__ import annotations

from .base import Topology
from .orchestrator_worker import OrchestratorWorker
from .peer_tool_graph import PeerToolGraph
from .single_react import SingleReAct

TOPOLOGIES: dict[str, type[Topology]] = {
    SingleReAct.name: SingleReAct,
    OrchestratorWorker.name: OrchestratorWorker,
    PeerToolGraph.name: PeerToolGraph,
}


def build_topology(name: str, llm, max_steps: int = 12) -> Topology:
    if name not in TOPOLOGIES:
        raise KeyError(f"unknown topology {name!r}; known: {sorted(TOPOLOGIES)}")
    return TOPOLOGIES[name](llm, max_steps=max_steps)
