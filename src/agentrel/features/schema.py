"""Feature schema (NEXT milestone — stub).

This file documents the features the modelling half will build from the labelled
corpus, so the boundary of the current ~30% is explicit. Implementations land in
the next milestone; the names here are the contract block (C) in the workflow.

STATIC (from the input data)          report 3.4
    rows, columns, frac_missing, frac_categorical, max_cardinality,
    mean_abs_skew, correlation_density, question_kind, n_tables
DYNAMIC (from the execution trace)
    n_tool_calls, n_retries, n_errors_recovered, code_lines, code_ast_nodes,
    subagent_handoffs, latency_s, tokens_used
GRAPH (from the tool-interaction graph G=(V,E))
    n_cycles, transition_entropy, max_betweenness

Transforms: log1p on counts, one-hot on categoricals, standardise numerics.
"""
from __future__ import annotations

STATIC_FEATURES = [
    "n_rows", "n_cols", "frac_missing", "frac_categorical", "max_cardinality",
    "mean_abs_skew", "correlation_density", "question_kind", "n_tables",
]
DYNAMIC_FEATURES = [
    "n_tool_calls", "n_retries", "n_errors", "n_llm_calls",
    "prompt_tokens", "completion_tokens", "wall_time_s",
]
GRAPH_FEATURES = [
    "n_cycles", "transition_entropy", "max_betweenness",
]

ALL_FEATURES = STATIC_FEATURES + DYNAMIC_FEATURES + GRAPH_FEATURES


def not_implemented(*_a, **_k):  # pragma: no cover - placeholder
    raise NotImplementedError(
        "feature engineering is the next milestone; see src/agentrel/features/schema.py")
