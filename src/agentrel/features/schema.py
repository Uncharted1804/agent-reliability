"""Feature schema — the contract block (C) in the workflow.

The names here ARE the contract: every block below is implemented in this
package and each implementation asserts it produced exactly these keys, so a
feature added to one and forgotten in the other fails loudly.

STATIC (from the input data)          report 3.4 -> static.py
    n_rows, n_cols, frac_missing, frac_categorical, max_cardinality,
    mean_abs_skew, correlation_density, question_kind, n_tables
DYNAMIC (from the execution trace)               -> dynamic.py
    n_tool_calls, n_retries, n_errors, n_llm_calls,
    prompt_tokens, completion_tokens, wall_time_s
GRAPH (from the tool-interaction graph G=(V,E))  -> graph.py
    n_cycles, transition_entropy, max_betweenness

Transforms: log1p on counts, one-hot on categoricals, standardise numerics.
They live in the model pipeline (model/pipeline.py) rather than here, so they
are fitted on training folds only and the raw frame stays inspectable.
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

