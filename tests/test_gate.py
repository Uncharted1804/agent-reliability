"""The silent-failure gate (report Algorithm 3) on a synthetic frame."""
import math

from test_relate import _frame

from agentrel.gate import Gate, fit_gate


def test_fit_gate_picks_a_threshold_from_out_of_fold_scores():
    gate = fit_gate(_frame(n_templates=40))
    assert 0 < gate.threshold < 1


def test_decide_holds_exactly_the_runs_at_or_above_threshold():
    df = _frame(n_templates=40)
    gate = fit_gate(df)
    out = gate.decide(df)
    assert set(out["decision"]) <= {"hold", "release"}
    assert ((out["p_failure"] >= gate.threshold) == (out["decision"] == "hold")).all()


def test_budget_mode_holds_the_top_k_share():
    df = _frame(n_templates=40)
    out = fit_gate(df).decide(df, budget=0.2)
    held = out[out["decision"] == "hold"]
    assert len(held) == math.ceil(0.2 * len(df))
    assert held["p_failure"].min() >= out[out["decision"] == "release"]["p_failure"].max()


def test_gate_round_trips_through_a_file(tmp_path):
    df = _frame(n_templates=40)
    gate = fit_gate(df)
    gate.save(tmp_path / "gate.joblib")
    again = Gate.load(tmp_path / "gate.joblib")
    assert again.threshold == gate.threshold
    assert (again.decide(df)["p_failure"] == gate.decide(df)["p_failure"]).all()
