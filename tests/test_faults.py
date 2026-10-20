"""Unit tests for fault injection and the robustness estimator.

The two things that would quietly corrupt Delta_{T,f} are tested directly:
counting unrealised faults as faulted (biases Delta toward zero), and
bootstrapping over runs instead of templates (CIs too narrow because
paraphrases are near duplicates, not independent observations).
"""
import numpy as np
import pandas as pd
import pytest

from agentrel.agents.tools import Session, ToolResult
from agentrel.faults.inject import (FAULTS, FaultInjector, _seeded_target, corrupt_value)
from agentrel.faults.robustness import (bootstrap_ci, delta, mean_degradation, prepare,
                                        robustness_table)


def _session():
    return Session(tables={}, scratch={"last_value": 100.0})


def _injector(fault, run_id="titanic.q01.p0::single_react::f"):
    return FaultInjector(fault=fault, run_id=run_id)


def _ok(value):
    return ToolResult(ok=True, value=value, repr=repr(value))


# --- injector basics ------------------------------------------------------- #
def test_rejects_unknown_fault():
    with pytest.raises(ValueError, match="unknown fault"):
        FaultInjector(fault="cosmic_ray", run_id="r")


def test_no_fault_is_a_passthrough():
    inj = FaultInjector(fault=None, run_id="r")
    result = _ok(5.0)
    assert inj.on_tool_result(0, "aggregate", result, _session()) is result
    assert inj.on_prompt("Step 0: aggregate -> OK 5.0") == "Step 0: aggregate -> OK 5.0"
    assert inj.realised is False


def test_target_step_is_deterministic_per_run_and_fault():
    # reproducibility: the same run faults identically on re-execution
    assert _seeded_target("r1", "tool_failure") == _seeded_target("r1", "tool_failure")
    targets = {_seeded_target(f"run{i}", "tool_failure") for i in range(40)}
    assert len(targets) > 1          # not a constant across runs


def test_target_step_within_window():
    assert all(0 <= _seeded_target(f"r{i}", f) < 3
               for i in range(30) for f in FAULTS)


# --- tool_failure ---------------------------------------------------------- #
def test_tool_failure_turns_a_good_result_into_an_error():
    inj = _injector("tool_failure")
    out = inj.on_tool_result(inj.target_step, "aggregate", _ok(5.0), _session())
    assert out.ok is False
    assert "InjectedToolFailure" in out.error
    assert inj.realised is True


def test_fault_fires_only_at_its_target_step():
    inj = _injector("tool_failure")
    other = (inj.target_step + 1) % 3
    out = inj.on_tool_result(other, "aggregate", _ok(5.0), _session())
    assert out.ok is True
    assert inj.realised is False


def test_fault_fires_at_most_once():
    inj = _injector("tool_failure")
    inj.on_tool_result(inj.target_step, "aggregate", _ok(5.0), _session())
    again = inj.on_tool_result(inj.target_step, "aggregate", _ok(7.0), _session())
    assert again.ok is True          # second call passes through untouched


# --- corrupted_output ------------------------------------------------------ #
def test_corruption_changes_the_value_but_keeps_it_ok():
    inj = _injector("corrupted_output")
    out = inj.on_tool_result(inj.target_step, "aggregate", _ok(100.0), _session())
    assert out.ok is True            # silent: nothing signals a problem
    assert out.value != 100.0
    assert inj.realised is True


def test_corruption_breaks_the_labelling_tolerance():
    # a corruption inside the 1% tolerance would not change the label at all
    assert abs(corrupt_value(100.0) - 100.0) / 100.0 > 0.01


def test_corruption_propagates_into_the_session_scratch():
    inj = _injector("corrupted_output")
    session = _session()
    inj.on_tool_result(inj.target_step, "aggregate", _ok(100.0), session)
    assert session.scratch["last_value"] != 100.0


def test_corruption_preserves_types():
    assert isinstance(corrupt_value(10), int)
    assert isinstance(corrupt_value(10.0), float)
    assert corrupt_value(True) is False
    assert corrupt_value("home") == "home"      # no plausible silent mutation


def test_corruption_perturbs_numeric_dict_entries_only():
    out = corrupt_value({"rows_after": 100, "loaded": "main"})
    assert out["rows_after"] != 100
    assert out["loaded"] == "main"


def test_corruption_skips_an_already_failed_result():
    inj = _injector("corrupted_output")
    failed = ToolResult(ok=False, error="KeyError: x")
    out = inj.on_tool_result(inj.target_step, "aggregate", failed, _session())
    assert out is failed
    assert inj.realised is False     # nothing was corrupted, so nothing fired


# --- dropped_context ------------------------------------------------------- #
def test_dropped_context_redacts_a_step_result():
    inj = FaultInjector(fault="dropped_context", run_id="r", window=1)
    prompt = "Question: q\nStep 0: aggregate({}) -> OK 0.38\n"
    out = inj.on_prompt(prompt)
    assert "0.38" not in out
    assert "[result unavailable]" in out
    assert inj.realised is True


def test_dropped_context_waits_until_a_result_exists():
    inj = FaultInjector(fault="dropped_context", run_id="r", window=1)
    assert inj.on_prompt("Question: q\n") == "Question: q\n"
    assert inj.realised is False


def test_dropped_context_leaves_other_steps_intact():
    inj = FaultInjector(fault="dropped_context", run_id="r", window=1)
    prompt = ("Question: q\nStep 0: load_table({}) -> OK rows=887\n"
              "Step 1: aggregate({}) -> OK 0.38\n")
    out = inj.on_prompt(prompt)
    assert "rows=887" in out         # the earlier step is still visible
    assert "0.38" not in out         # the most recent result is withheld


def test_describe_records_provenance_for_the_trace():
    inj = _injector("tool_failure")
    inj.on_tool_result(inj.target_step, "aggregate", _ok(1.0), _session())
    d = inj.describe()
    assert d["injected_fault"] == "tool_failure"
    assert d["fault_realised"] is True
    assert d["fault_target_step"] == inj.target_step


# --- robustness ------------------------------------------------------------ #
def _corpus(topologies=("single_react", "peer_tool_graph"), n_templates=12,
            paraphrases=3, base_rate=0.2, fault_rate=0.6, seed=0):
    """Matched baseline + faulted labelled frames."""
    rng = np.random.default_rng(seed)
    base, flt = [], []
    for t in range(n_templates):
        tid = f"ds.q{t:02d}"
        for p in range(paraphrases):
            qid = f"{tid}.p{p}"
            for topo in topologies:
                base.append({"run_id": f"{qid}::{topo}", "question_id": qid,
                             "template_id": tid, "topology": topo,
                             "injected_fault": None, "fault_realised": 0,
                             "label_failure": int(rng.random() < base_rate)})
                for fault in ("tool_failure", "corrupted_output"):
                    flt.append({"run_id": f"{qid}::{topo}::{fault}", "question_id": qid,
                                "template_id": tid, "topology": topo,
                                "injected_fault": fault, "fault_realised": 1,
                                "label_failure": int(rng.random() < fault_rate)})
    return pd.DataFrame(base), pd.DataFrame(flt)


def test_prepare_drops_unrealised_faults():
    base, flt = _corpus()
    flt.loc[flt.index[:10], "fault_realised"] = 0
    _, kept, info = prepare(base, flt)
    assert len(kept) == len(flt) - 10
    assert info["fault_realisation_rate"] == pytest.approx(1 - 10 / len(flt))
    assert (kept["fault_realised"] == 1).all()


def test_prepare_restricts_baseline_to_matched_questions():
    base, flt = _corpus()
    flt = flt[flt["question_id"] != "ds.q00.p0"]
    matched_base, _, info = prepare(base, flt)
    assert "ds.q00.p0" not in set(matched_base["question_id"])
    assert info["n_matched_questions"] == base["question_id"].nunique() - 1


def test_delta_is_the_failure_rate_increase():
    base = pd.DataFrame({"label_failure": [0, 0, 0, 0]})
    flt = pd.DataFrame({"label_failure": [1, 1, 0, 0]})
    assert delta(base, flt) == pytest.approx(0.5)


def test_delta_is_zero_for_a_neutralised_fault():
    base = pd.DataFrame({"label_failure": [1, 0, 1, 0]})
    assert delta(base, base) == 0.0


def test_robustness_table_has_one_row_per_topology_fault_cell():
    base, flt = _corpus()
    table = robustness_table(base, flt, n_resamples=200)
    assert len(table) == 2 * 2          # 2 topologies x 2 faults
    assert set(table.columns) >= {"topology", "fault", "delta", "ci_lo", "ci_hi",
                                  "significant"}
    assert (table["ci_lo"] <= table["delta"]).all()
    assert (table["delta"] <= table["ci_hi"]).all()


def test_bootstrap_detects_a_real_degradation():
    base, flt = _corpus(base_rate=0.05, fault_rate=0.85, n_templates=20)
    table = robustness_table(base, flt, n_resamples=500)
    assert (table["delta"] > 0.5).all()
    assert table["significant"].all()       # CI excludes zero
    assert (table["ci_lo"] > 0).all()


def test_bootstrap_ci_spans_zero_for_a_perfectly_neutralised_fault():
    # faulted labels identical to baseline: Delta is exactly 0 by construction,
    # so the interval must cover 0 and the cell must not read as significant
    base, flt = _corpus(n_templates=16)
    paired = base.set_index(["question_id", "topology"])["label_failure"]
    flt = flt.copy()
    flt["label_failure"] = [paired[(q, t)] for q, t in
                            zip(flt["question_id"], flt["topology"])]
    table = robustness_table(base, flt, n_resamples=500)
    assert (table["delta"].abs() < 1e-12).all()
    assert (table["ci_lo"] <= 0).all() and (table["ci_hi"] >= 0).all()
    assert not table["significant"].any()


def test_bootstrap_ci_covers_the_truth_under_the_null():
    # calibration: with no real effect the 95% interval should cover 0 for the
    # large majority of corpora. Individual corpora can still look significant
    # from sampling noise alone, which is why this is a rate, not an assertion
    # about one draw.
    covered = 0
    trials = 20
    for seed in range(trials):
        base, flt = _corpus(base_rate=0.4, fault_rate=0.4, n_templates=20, seed=seed)
        lo, hi, _ = bootstrap_ci(base, flt, n_resamples=200, seed=seed)
        covered += int(lo <= 0 <= hi)
    assert covered >= 0.80 * trials


def test_bootstrap_is_reproducible_for_a_seed():
    base, flt = _corpus()
    a = bootstrap_ci(base, flt, n_resamples=200, seed=1)
    b = bootstrap_ci(base, flt, n_resamples=200, seed=1)
    assert a == b
    assert bootstrap_ci(base, flt, n_resamples=200, seed=2) != a


def test_cluster_bootstrap_is_wider_than_resampling_runs():
    # paraphrases of one template are near duplicates: treating them as
    # independent runs understates uncertainty, so the cluster CI must be wider
    base, flt = _corpus(n_templates=10, paraphrases=5)
    lo_c, hi_c, _ = bootstrap_ci(base, flt, n_resamples=400, cluster="template_id")
    lo_r, hi_r, _ = bootstrap_ci(base, flt, n_resamples=400, cluster="run_id")
    assert (hi_c - lo_c) > (hi_r - lo_r)


def test_mean_degradation_ranks_topologies_most_robust_first():
    base, flt = _corpus()
    flt.loc[flt["topology"] == "single_react", "label_failure"] = 1
    ranked = mean_degradation(robustness_table(base, flt, n_resamples=100))
    assert ranked.iloc[-1]["topology"] == "single_react"    # worst is last
