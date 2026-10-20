"""Unit tests for the three feature blocks.

The graph features are the ones worth testing hardest: they are the only block
whose value depends on the *order* of the trace rather than on counts, so a bug
there is invisible in a summary table. Each case below is a trace shape the
topologies actually produce.
"""
import math

import numpy as np
import pandas as pd
import pytest

from agentrel.features.dynamic import dynamic_features
from agentrel.features.graph import (build_graph, graph_features, max_betweenness,
                                     n_cycles, transition_entropy, tool_sequence)
from agentrel.features.schema import DYNAMIC_FEATURES, GRAPH_FEATURES, STATIC_FEATURES
from agentrel.features.static import (correlation_density, mean_abs_skew, static_features,
                                      table_features)


def _run(tools, **kw):
    """Minimal run dict with a tool sequence (ok unless marked otherwise)."""
    calls = []
    for i, t in enumerate(tools):
        call = {"step": i, "agent_role": "react", "tool": t, "args": {}, "ok": True,
                "is_retry": False}
        call.update(kw.pop(f"call{i}", {}))
        calls.append(call)
    run = {"run_id": "r", "dataset": "titanic", "template_id": "titanic.q01",
           "calls": calls}
    run.update(kw)
    return run


# --- static ---------------------------------------------------------------- #
def test_table_features_shape_and_missingness():
    df = pd.DataFrame({"a": [1.0, 2.0, None, 4.0], "b": ["x", "y", "x", "z"]})
    f = table_features(df)
    assert f["n_rows"] == 4 and f["n_cols"] == 2
    assert f["frac_missing"] == pytest.approx(1 / 8)      # one missing cell of eight
    assert f["frac_categorical"] == pytest.approx(0.5)    # b is not numeric
    assert f["max_cardinality"] == 3                      # a has 3 non-null distinct


def test_correlation_density_detects_perfect_correlation():
    df = pd.DataFrame({"a": [1.0, 2, 3, 4, 5], "b": [2.0, 4, 6, 8, 10]})
    assert correlation_density(df) == 1.0                  # the single pair clears 0.3


def test_correlation_density_zero_when_too_few_numeric_columns():
    assert correlation_density(pd.DataFrame({"a": [1, 2, 3]})) == 0.0


def test_mean_abs_skew_positive_for_skewed_column():
    skewed = pd.DataFrame({"a": [1.0] * 20 + [500.0]})
    symmetric = pd.DataFrame({"a": list(np.linspace(-1, 1, 21))})
    assert mean_abs_skew(skewed) > mean_abs_skew(symmetric)


def test_mean_abs_skew_ignores_constant_columns():
    # a constant column has undefined skew; it must not poison the mean
    assert mean_abs_skew(pd.DataFrame({"a": [5.0] * 10})) == 0.0


def test_static_features_cover_the_contract():
    f = static_features("titanic", "aggregation", 1)
    assert set(f) == set(STATIC_FEATURES)
    assert f["question_kind"] == "aggregation"


def test_static_features_identical_across_topologies():
    # the static block describes the task, so it must not vary by run
    assert static_features("titanic", "aggregation", 1) == \
           static_features("titanic", "aggregation", 1)


# --- dynamic --------------------------------------------------------------- #
def test_dynamic_features_cover_the_contract():
    run = _run(["load_table", "aggregate"], n_llm_calls=3, prompt_tokens=120,
               completion_tokens=18, wall_time_s=7.6)
    f = dynamic_features(run)
    assert set(f) == set(DYNAMIC_FEATURES)
    assert f["n_llm_calls"] == 3 and f["wall_time_s"] == pytest.approx(7.6)


def test_dynamic_counts_recomputed_when_derived_fields_absent():
    run = _run(["aggregate", "aggregate"], call1={"ok": False, "is_retry": True})
    f = dynamic_features(run)
    assert f["n_tool_calls"] == 2
    assert f["n_errors"] == 1
    assert f["n_retries"] == 1


def test_dynamic_prefers_derived_fields_when_present():
    run = _run(["aggregate"], n_tool_calls=9, n_errors=4, n_retries=2)
    f = dynamic_features(run)
    assert (f["n_tool_calls"], f["n_errors"], f["n_retries"]) == (9.0, 4.0, 2.0)


def test_dynamic_handles_a_run_with_no_calls():
    f = dynamic_features({"run_id": "r"})
    assert f["n_tool_calls"] == 0 and f["wall_time_s"] == 0.0


# --- graph ----------------------------------------------------------------- #
def test_tool_sequence_skips_unresolved_tools():
    # an unparsed action records an empty tool name; it is not a graph node
    run = _run(["load_table", "", "aggregate"])
    assert tool_sequence(run) == ["load_table", "aggregate"]


def test_graph_edges_are_weighted_transitions():
    g = build_graph(["a", "b", "a", "b"])
    assert set(g.nodes) == {"a", "b"}
    assert g["a"]["b"]["weight"] == 2      # a->b happened twice
    assert g["b"]["a"]["weight"] == 1


def test_linear_pass_has_no_cycles():
    g = build_graph(["load_table", "filter_rows", "aggregate"])
    assert n_cycles(g) == 0


def test_oscillation_is_a_cycle():
    # agent bouncing between two tools: a->b->a
    assert n_cycles(build_graph(["a", "b", "a"])) == 1


def test_consecutive_repeat_is_a_self_loop_cycle():
    # a retry of the same tool is a length-1 cycle
    assert n_cycles(build_graph(["aggregate", "aggregate"])) == 1


def test_transition_entropy_zero_when_every_move_is_the_same():
    assert transition_entropy(["a", "b", "a", "b", "a"]) > 0        # two distinct moves
    assert transition_entropy(["a", "a", "a"]) == 0.0              # only a->a


def test_transition_entropy_matches_hand_computation():
    # moves: a->b, b->c  => two equally likely transitions => 1 bit
    assert transition_entropy(["a", "b", "c"]) == pytest.approx(1.0)
    # four distinct equally likely transitions => 2 bits
    assert transition_entropy(["a", "b", "c", "d", "e"]) == pytest.approx(2.0)


def test_max_betweenness_zero_for_graphs_without_through_paths():
    assert max_betweenness(build_graph(["a"])) == 0.0
    assert max_betweenness(build_graph(["a", "b"])) == 0.0


def test_max_betweenness_identifies_a_hinge_tool():
    # a -> b -> c : b is the only node lying between the others
    g = build_graph(["a", "b", "c"])
    assert max_betweenness(g) > 0.0


def test_graph_features_cover_the_contract_and_zero_on_empty_trace():
    f = graph_features(_run([]))
    assert set(f) == set(GRAPH_FEATURES)
    assert all(v == 0.0 for v in f.values())      # answered without doing any work


def test_graph_features_on_a_realistic_retry_trace():
    # the shape the topologies emit when a tool errors and is retried
    f = graph_features(_run(["load_table", "aggregate", "aggregate", "group_aggregate"]))
    assert f["n_cycles"] == 1                     # the aggregate self-loop
    assert f["transition_entropy"] == pytest.approx(math.log2(3))
    assert f["max_betweenness"] > 0.0
