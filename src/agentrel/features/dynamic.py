"""Dynamic features — properties of the execution trace (report 3.4).

These describe *how the agent behaved* on one run: how much tool work it did,
how often it stumbled and recovered, how much the model was asked, and how long
it took. Unlike the static block these differ across topologies for the same
question, which is what lets the classifier separate "hard task" from
"agent went wrong".

Read straight off the raw run dict so the feature layer depends on the trace
schema (harness/trace.py) rather than on the labelled table, keeping block (C)
independent of block (B).
"""
from __future__ import annotations

from typing import Any

from .schema import DYNAMIC_FEATURES


def dynamic_features(run: dict[str, Any]) -> dict[str, float]:
    """Trace-level numeric features for one run.

    Counts are taken from the trace's own derived fields when present and
    recomputed from `calls` otherwise, so a trace written by an older harness
    version still yields the full block.
    """
    calls = run.get("calls") or []

    def _count(key: str, predicate) -> float:
        if key in run and run[key] is not None:
            return float(run[key])
        return float(sum(1 for c in calls if predicate(c)))

    feats = {
        "n_tool_calls": _count("n_tool_calls", lambda c: True),
        "n_retries": _count("n_retries", lambda c: c.get("is_retry")),
        "n_errors": _count("n_errors", lambda c: not c.get("ok", True)),
        "n_llm_calls": float(run.get("n_llm_calls", 0) or 0),
        "prompt_tokens": float(run.get("prompt_tokens", 0) or 0),
        "completion_tokens": float(run.get("completion_tokens", 0) or 0),
        "wall_time_s": float(run.get("wall_time_s", 0.0) or 0.0),
    }
    missing = set(DYNAMIC_FEATURES) - set(feats)
    if missing:  # pragma: no cover — guards the schema contract
        raise AssertionError(f"dynamic features missing from schema contract: {missing}")
    return feats
