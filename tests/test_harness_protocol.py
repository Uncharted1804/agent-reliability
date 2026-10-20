"""Unit tests for the action protocol and the schema digest.

A model that deviates from the protocol must still produce a usable trace: if
an action fails to parse, the run records no tool calls, and the dynamic and
graph features then describe the parser rather than the agent. Every deviation
asserted here was observed from a real local model.
"""
import pandas as pd

from agentrel.agents.tools import Session, schema_digest
from agentrel.harness.trace import RunTrace
from agentrel.topology.base import initial_history, normalise_action, parse_action


def _session() -> Session:
    return Session(tables={"main": pd.DataFrame({
        "age": [30.0, 40.0, None],
        "sex": ["male", "female", "male"],
        "paid": [1, 0, 1],
    })})


# --- action parsing -------------------------------------------------------- #
def test_canonical_tool_action():
    got = parse_action('{"action":"tool","tool":"aggregate","args":{"op":"mean","column":"age"}}')
    assert got == {"action": "tool", "tool": "aggregate",
                   "args": {"op": "mean", "column": "age"}}


def test_tool_name_used_as_action_verb():
    # observed: {"action":"load_table", ...} instead of action="tool"
    got = parse_action('{"action":"load_table","args":{"table":"main"}}')
    assert got == {"action": "tool", "tool": "load_table", "args": {"table": "main"}}


def test_args_flattened_to_top_level():
    # observed: arguments alongside the action rather than nested under "args"
    got = parse_action('{"action":"aggregate","op":"mean","column":"paid"}')
    assert got == {"action": "tool", "tool": "aggregate",
                   "args": {"op": "mean", "column": "paid"}}


def test_multi_action_plan_takes_first_step_only():
    # observed: a whole plan emitted at once; the loop re-prompts for the rest
    text = ('{"action":"load_table","args":{"table":"main"}}\n'
            '{"action":"aggregate","op":"mean","column":"paid"}\n'
            '{"action":"final","answer":"0.66"}')
    assert parse_action(text) == {"action": "tool", "tool": "load_table",
                                  "args": {"table": "main"}}


def test_final_action_numeric_and_string():
    assert parse_action('{"action":"final","answer":0.38}')["answer"] == 0.38
    assert parse_action('{"action":"final","answer":"home"}')["answer"] == "home"


def test_control_actions_survive():
    # the inspector and peer-handoff verdicts must not be mangled into tools
    for act in ("accept", "redo", "handoff"):
        assert parse_action('{"action":"%s"}' % act) == {"action": act}


def test_json_embedded_in_prose():
    got = parse_action('Sure, here is my step:\n{"action":"tool","tool":"list_tables","args":{}}\nDone.')
    assert got["tool"] == "list_tables"


def test_unparseable_reply_becomes_final_answer():
    # keeps the run in the corpus as a likely failure instead of dropping it
    got = parse_action("The survival rate is about 0.38")
    assert got["action"] == "final"
    assert "0.38" in got["answer"]


def test_unknown_object_is_not_a_tool_call():
    assert normalise_action({"thinking": "let me see"}) == {}


# --- schema digest --------------------------------------------------------- #
def test_digest_reports_shape_and_dtypes():
    text = schema_digest(_session())
    assert "main: 3 rows, 3 columns" in text
    assert "age (float64" in text


def test_digest_reports_missing_rate_and_low_cardinality_values():
    text = schema_digest(_session())
    assert "33% missing" in text                     # one of three ages absent
    assert "values: ['female', 'male']" in text      # agent needs the labels to filter


def test_digest_omits_missing_note_for_complete_column():
    assert "paid (int64)" in schema_digest(_session())


def test_initial_history_carries_schema_and_question_but_not_answers():
    trace = RunTrace(run_id="r", dataset="d", template_id="t", question_id="q",
                     paraphrase_index=0, topology="single_react", model="m",
                     question_text="What fraction paid?")
    text = initial_history(trace, _session())
    assert "What fraction paid?" in text
    assert "main: 3 rows" in text
    # descriptive only: no computed aggregate leaks into the preamble
    assert "0.66" not in text
