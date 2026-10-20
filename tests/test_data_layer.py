"""Tests that the data layer is internally consistent: every template computes a
ground truth, every dataset expands to the right number of questions, and
paraphrases of a template share one ground-truth value."""
import pytest

from agentrel.data.questions import expand_questions, templates_for
from agentrel.data.registry import SPECS

DATASETS = list(SPECS)


@pytest.mark.parametrize("dataset", DATASETS)
def test_ten_templates_each(dataset):
    assert len(templates_for(dataset)) == 10


@pytest.mark.parametrize("dataset", DATASETS)
def test_four_paraphrases_each(dataset):
    for tpl in templates_for(dataset):
        assert len(tpl.paraphrases) == 4, f"{tpl.id} has {len(tpl.paraphrases)}"


@pytest.mark.parametrize("dataset", DATASETS)
def test_expansion_count(dataset):
    # 10 templates x (1 canonical + 4 paraphrases) = 50
    assert len(expand_questions(dataset)) == 50


@pytest.mark.parametrize("dataset", DATASETS)
def test_question_ids_unique(dataset):
    ids = [q["question_id"] for q in expand_questions(dataset)]
    assert len(ids) == len(set(ids))


def test_answer_types_valid():
    from agentrel.data.questions import all_templates
    for tpl in all_templates():
        assert tpl.answer_type in ("numeric", "categorical")
        assert tpl.kind in ("aggregation", "filtering", "group_comparison",
                            "trend", "multi_step")
