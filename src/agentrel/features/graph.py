"""Graph features — the tool-interaction graph G=(V,E) (report 3.4).

One run's trace is a sequence of tool calls. Turning it into a directed graph
exposes *shape* that counting cannot: a run that oscillates between two tools,
or funnels everything through one bottleneck tool, looks different from a clean
linear pass even when both make the same number of calls.

    V = the distinct tools the run invoked
    E = observed transitions tool_i -> tool_{i+1}, weighted by how often the
        agent made that transition (consecutive repeats become self-loops)

The three features are the ones named in the schema contract:

    n_cycles           number of simple directed cycles (self-loops included) --
                       retry loops and oscillation between two tools
    transition_entropy Shannon entropy (bits) of the transition distribution --
                       0 when the run always makes the same move, high when the
                       agent's next step is unpredictable
    max_betweenness    largest betweenness centrality over V -- how much one
                       tool acts as the single hinge the run passes through

A run with fewer than two calls has no transitions, so all three are 0: a
degenerate graph is itself informative (the agent answered without working).
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable

import networkx as nx

from .schema import GRAPH_FEATURES

# simple_cycles is exponential in the worst case; a trace is capped at
# max_agent_steps calls so the graph is tiny, but the bound keeps a pathological
# trace from stalling the feature build.
MAX_CYCLES = 1000


def tool_sequence(run: dict[str, Any]) -> list[str]:
    """Ordered tool names for a run, skipping calls with no resolved tool."""
    return [c.get("tool") or "" for c in (run.get("calls") or []) if c.get("tool")]


def build_graph(tools: Iterable[str]) -> nx.DiGraph:
    """Directed, weighted tool-interaction graph for one run."""
    tools = list(tools)
    g = nx.DiGraph()
    g.add_nodes_from(tools)
    transitions = Counter(zip(tools, tools[1:]))
    for (src, dst), weight in transitions.items():
        g.add_edge(src, dst, weight=int(weight))
    return g


def transition_entropy(tools: Iterable[str]) -> float:
    """Shannon entropy (bits) of the empirical transition distribution."""
    tools = list(tools)
    transitions = Counter(zip(tools, tools[1:]))
    total = sum(transitions.values())
    if total == 0:
        return 0.0
    return float(-sum((c / total) * math.log2(c / total) for c in transitions.values()))


def n_cycles(g: nx.DiGraph) -> float:
    """Count simple directed cycles, including self-loops, capped at MAX_CYCLES."""
    count = 0
    for _ in nx.simple_cycles(g):
        count += 1
        if count >= MAX_CYCLES:
            break
    return float(count)


def max_betweenness(g: nx.DiGraph) -> float:
    """Largest node betweenness centrality; 0 for graphs too small to have paths."""
    if g.number_of_nodes() < 3:
        # with fewer than three nodes no node can lie *between* two others
        return 0.0
    centrality = nx.betweenness_centrality(g, normalized=True)
    return float(max(centrality.values(), default=0.0))


def graph_features(run: dict[str, Any]) -> dict[str, float]:
    """Graph feature block for one run."""
    tools = tool_sequence(run)
    g = build_graph(tools)
    feats = {
        "n_cycles": n_cycles(g),
        "transition_entropy": transition_entropy(tools),
        "max_betweenness": max_betweenness(g),
    }
    missing = set(GRAPH_FEATURES) - set(feats)
    if missing:  # pragma: no cover — guards the schema contract
        raise AssertionError(f"graph features missing from schema contract: {missing}")
    return feats
