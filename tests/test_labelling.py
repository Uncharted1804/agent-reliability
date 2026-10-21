"""Unit tests for the labelling + parsing logic — the part that must be exactly
right regardless of which LLM produced the runs."""
from agentrel.labelling.label import label_run, paraphrase_consistency
from agentrel.labelling.parse import parse_categorical, parse_numeric


def _run(ans, rid="t.q01.p0", tid="t.q01", topo="single_react", pidx=0):
    return {"run_id": rid, "dataset": "d", "template_id": tid, "question_id": rid,
            "topology": topo, "paraphrase_index": pidx, "final_answer_raw": ans,
            "timed_out": False, "harness_error": None}


# --- numeric parsing ------------------------------------------------------- #
def test_parse_plain_number():
    assert parse_numeric("The answer is 0.3625") == 0.3625


def test_parse_percentage_to_fraction():
    # no bare fraction present -> 38% becomes 0.38
    assert parse_numeric("about 38%") == 0.38


def test_parse_prefers_fraction_over_percent():
    # a plain fraction is present, so don't convert the percent
    assert parse_numeric("the rate 0.38 (i.e. 38%)") == 0.38


def test_parse_handles_thousands_separator():
    assert parse_numeric("total is 12,345") == 12345.0


def test_parse_integer_without_separator():
    # seen live: "32561" parsed as 61 and "2000" as 0, mislabelling correct runs
    assert parse_numeric("32561") == 32561.0
    assert parse_numeric("2000") == 2000.0
    assert parse_numeric("-1500.25") == -1500.25


def test_parse_none_on_empty():
    assert parse_numeric("") is None
    assert parse_numeric(None) is None


# --- categorical parsing --------------------------------------------------- #
def test_parse_categorical_normalises():
    assert parse_categorical('"Home."') == "home"
    assert parse_categorical("The answer is: Electronics") == "electronics"


# --- Eq.1 labelling -------------------------------------------------------- #
def test_numeric_within_tolerance_is_success():
    truth = {"answer_type": "numeric", "value": 100.0, "error": None}
    lab = label_run(_run("100.5"), truth, eps=0.01)  # 0.5% < 1%
    assert lab["label_failure"] == 0


def test_numeric_outside_tolerance_is_failure():
    truth = {"answer_type": "numeric", "value": 100.0, "error": None}
    lab = label_run(_run("150"), truth, eps=0.01)
    assert lab["label_failure"] == 1


def test_categorical_exact_match_is_success():
    truth = {"answer_type": "categorical", "value": "home", "error": None}
    assert label_run(_run("Home"), truth, eps=0.01)["label_failure"] == 0


def test_categorical_mismatch_is_failure():
    truth = {"answer_type": "categorical", "value": "home", "error": None}
    assert label_run(_run("electronics"), truth, eps=0.01)["label_failure"] == 1


def test_unparseable_answer_is_failure():
    truth = {"answer_type": "numeric", "value": 100.0, "error": None}
    lab = label_run(_run("I could not compute this"), truth, eps=0.01)
    assert lab["label_failure"] == 1
    assert lab["label_reason"] == "unparseable_answer"


def test_ground_truth_error_is_failure():
    truth = {"answer_type": "numeric", "value": None, "error": "KeyError: x"}
    assert label_run(_run("3"), truth, eps=0.01)["label_failure"] == 1


# --- paraphrase consistency ------------------------------------------------ #
def test_paraphrase_consistency_agree():
    labs = [{"template_id": "t.q01", "topology": "single_react",
             "parsed_answer": 0.5} for _ in range(4)]
    out = paraphrase_consistency(labs)
    assert out[("t.q01", "single_react")] is True


def test_paraphrase_consistency_disagree():
    labs = [{"template_id": "t.q01", "topology": "single_react",
             "parsed_answer": v} for v in (0.5, 0.5, 0.9, 0.5)]
    out = paraphrase_consistency(labs)
    assert out[("t.q01", "single_react")] is False

# --- labelled-output paths ------------------------------------------------- #
def test_fault_corpus_labels_do_not_overwrite_baseline_labels():
    """The robustness study needs both tables, so their paths must differ."""
    from agentrel.labelling.pipeline import default_label_path

    baseline = default_label_path("results/runs.jsonl")
    faulted = default_label_path("results/runs_faults.jsonl")
    assert baseline.name == "labelled.parquet"      # the documented M1 path
    assert faulted != baseline
