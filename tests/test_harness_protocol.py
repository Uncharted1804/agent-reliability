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
def test_digest_names_tables_columns_and_dtypes():
    text = schema_digest(_session())
    assert "table main: columns ['age', 'sex', 'paid']" in text
    assert "age (float64" in text


def test_digest_flags_missingness_without_quantifying_it():
    # the missing RATE is the answer to the missing-rate templates, so the
    # agent is told only that the column has gaps
    text = schema_digest(_session())
    assert "has missing values" in text
    assert "33%" not in text and "% missing" not in text


def test_digest_gives_level_names_but_not_their_count():
    # the names are the option set a filter needs; the count is a cardinality answer
    text = schema_digest(_session())
    assert "values: ['female', 'male']" in text
    assert "distinct" not in text


def test_digest_omits_missing_note_for_complete_column():
    assert "paid (int64)" in schema_digest(_session())


def test_digest_states_no_row_count():
    # the row count is the literal answer to the count templates
    text = schema_digest(_session())
    assert "3 rows" not in text
    assert "rows" not in text


def test_digest_leaks_no_aggregate_of_the_data():
    """Regression: no quantity derived from the values may appear.

    Six templates were previously answerable straight off the preamble
    because it printed the row count, the missing rate and the distinct
    count. Any number in the digest is now a dtype artefact at most.
    """
    import re
    text = schema_digest(_session())
    numbers = {n for n in re.findall(r"(?<![\w.])(\d+)(?![\w.])", text)}
    # 3 rows, 2 non-null ages, 1 missing age, 2 sex levels, 2 paid -- none may show
    assert numbers.isdisjoint({"3", "1", "2"}), f"quantities leaked: {numbers}"


def test_initial_history_carries_schema_and_question_but_not_answers():
    trace = RunTrace(run_id="r", dataset="d", template_id="t", question_id="q",
                     paraphrase_index=0, topology="single_react", model="m",
                     question_text="What fraction paid?")
    text = initial_history(trace, _session())
    assert "What fraction paid?" in text
    assert "table main: columns" in text
    # descriptive only: no computed aggregate leaks into the preamble
    assert "0.66" not in text


# --- quantile op ----------------------------------------------------------- #
# Without it the 1.5*IQR templates were unanswerable through the tools, so
# every model failed them and they measured the toolset, not the agent.
def test_aggregate_quantile_matches_pandas():
    from agentrel.agents.tools import call_tool
    s = Session(tables={"main": pd.DataFrame({"fare": [1.0, 2.0, 3.0, 4.0, 100.0]})})
    q1 = call_tool("aggregate", s, {"op": "quantile", "column": "fare", "q": 0.25})
    q3 = call_tool("aggregate", s, {"op": "quantile", "column": "fare", "q": 0.75})
    assert (q1.ok, q1.value) == (True, 2.0)
    assert (q3.ok, q3.value) == (True, 4.0)


def test_aggregate_quantile_needs_q_in_range():
    from agentrel.agents.tools import call_tool
    s = Session(tables={"main": pd.DataFrame({"fare": [1.0, 2.0]})})
    assert not call_tool("aggregate", s, {"op": "quantile", "column": "fare"}).ok
    assert not call_tool("aggregate", s, {"op": "quantile", "column": "fare", "q": 25}).ok


def test_flattened_quantile_argument_is_kept():
    got = parse_action('{"action":"aggregate","op":"quantile","column":"fare","q":0.25}')
    assert got["args"] == {"op": "quantile", "column": "fare", "q": 0.25}


def test_protocol_documents_quantile():
    from agentrel.topology.base import ACTION_PROTOCOL
    assert "quantile" in ACTION_PROTOCOL and '"q"' in ACTION_PROTOCOL


# --- base LLM payload ------------------------------------------------------ #
def test_think_flag_reaches_ollama_only_when_set(monkeypatch):
    from agentrel.agents import llm as llm_mod
    from agentrel.utils.config import LLMConfig

    sent = []

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "{}"}}

    monkeypatch.setattr(llm_mod.requests, "post",
                        lambda url, json=None, timeout=None: sent.append(json) or _Resp())
    llm_mod.OllamaLLM(LLMConfig(model="qwen3:8b", think=False)).complete("s", "u")
    llm_mod.OllamaLLM(LLMConfig(model="llama3.1:8b")).complete("s", "u")
    assert sent[0]["think"] is False
    assert "think" not in sent[1]   # unset means the model's own default
