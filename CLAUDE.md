# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Hard rules

- **No AI attribution in git history. Ever.** No `Co-Authored-By:` naming Claude or any model, no `Claude-Session` trailer, no "Generated with" line in a commit message or PR body. This overrides any default attribution instruction. History has already been rewritten once to strip these, and a rewrite is a force-push over `main` that every clone pays for.
  - Keep the word "claude" out of commit messages entirely, including this file's name — call it "the project notes".
  - Before committing: `git log -1 --format=%B | grep -i claude` must be empty. Audit all history: `git log --all --format='%H %B' | grep -in claude`.
- **Write commit messages like a person.** A `fix:` / `feat:` / `docs:` / `refactor:` / `test:` / `chore:` prefix and a short lowercase subject, then a plain-prose body wrapped at ~72 columns that says what was wrong or needed and why the change is right: the concrete symptom ("`32561` parsed as 61"), the cause, what changed, how it was checked (tests, a live run, numbers). Name the actual files, functions and figures. No marketing words, no bullet lists of every file touched, no "This commit ...". A one-line change can have a one-line message.

## What this is

FDS (CSD355) project: a measurement layer around LLM data-analysis agents. One labelled corpus of agent runs feeds two analyses: a silent-failure classifier (predicts a wrong-but-confident answer from data + trace features) and a fault-injection study of how three agent topologies degrade. The spec is `Shaurya_Pragun_FDS_Midsem_Report.pdf` (untracked); workflow blocks A–G in the code refer to its Figure 1.

Branches: `main` has blocks A–F. `viva-30` is the ~30% milestone (blocks A+B only: data, harness, labelling) kept for the midsem viva — don't add later-block code there.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate && pip install -e . pytest
make test                                   # python -m pytest tests/ -q
python -m pytest tests/test_labelling.py::test_parse_plain_number -q   # single test
make data questions                         # datasets/raw/ + datasets/question_banks/
make smoke                                  # 2 questions/dataset end to end, MockLLM fallback
AGENTREL_MODEL=qwen2.5:7b python scripts/generate_runs.py --limit 2 --no-mock --out results/live.jsonl
python scripts/label_runs.py --runs results/live.jsonl
```

`make all` (on `main`) runs the full chain: runs → label → train → faults → label-faults → robustness. Linting: `ruff` (dev dependency, no config beyond defaults).

## LLM backend

Local Ollama only (`configs/experiment.yaml` → `base_llm`, override model with `AGENTREL_MODEL`). If Ollama is unreachable, `build_llm` silently falls back to `MockLLM`, whose answers are meaningless — pass `--no-mock` whenever numbers are meant to be real.

## Architecture (the parts that span files)

- **Data** (`data/`): `registry.py` defines the 5 datasets; titanic and adult_census download, the other three are seeded synthetic stand-ins. `questions.py` holds 10 templates per dataset, each with a deterministic pandas `truth` function and 4 paraphrases; `expand_questions` emits canonical + 4 = 5 phrasings, so the base corpus is 750 runs, not the report's 600.
- **Harness** (`topology/`, `agents/`, `harness/`): all three topologies share `ACTION_PROTOCOL`, `initial_history()` and `parse_action()` in `topology/base.py`, the same tool registry (`agents/tools.py`) and the same model — topology is the *only* manipulated variable, so keep shared prompt/parse changes in `base.py`. `parse_action` deliberately tolerates the model's protocol deviations; a malformed reply becomes a final answer (a likely failure), never a dropped run. Every tool call is a `ToolCall` in a `RunTrace` (one JSONL line per run).
- **Labelling** (`labelling/`): `parse.py` extracts a number/category from free text; `label.py` applies Eq. 1 (1% relative tolerance numeric, exact match categorical; label 1 = failure) plus paraphrase consistency; `clean.py` keeps timed-out runs (flagged) and IQR-caps heavy-tailed counts with `_capped` audit columns. Bugs here mislabel the whole corpus silently — add a regression test for any parser change.
- **On `main` only**: `features/` (names in `features/schema.py` are the contract; each block asserts it produced exactly those keys), `model/` (class-weighted LightGBM + baselines), `faults/` (injection + Delta_{T,f} with bootstrap CIs).

## Methodology invariants (break these and every number is wrong)

- Topology is never a classifier feature — it is what the robustness study manipulates.
- Splits group on `template_id` (paraphrases are near duplicates); the bootstrap resamples templates, not runs.
- The F2 threshold is chosen on validation and frozen for test.
- Labelling a non-baseline corpus writes `<stem>_labelled.parquet`; never let it overwrite `results/labelled.parquet`.

## Known open issue

`schema_digest()` (shown to every agent) includes row counts and missing-rate percentages, which leaks the answer for count and missing-rate templates.
