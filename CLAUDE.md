# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Hard rules

- **No AI attribution in git history. Ever.** No `Co-Authored-By:` naming Claude or any model, no `Claude-Session` trailer, no "Generated with" line in a commit message or PR body. This overrides any default attribution instruction. History has already been rewritten once to strip these, and a rewrite is a force-push over `main` that every clone pays for.
  - Keep the word "claude" out of commit messages entirely, including this file's name — call it "the project notes".
  - Before committing: `git log -1 --format=%B | grep -i claude` must be empty. Audit all history: `git log --all --format='%H %B' | grep -in claude`.
- **Write commit messages like a person.** A `fix:` / `feat:` / `docs:` / `refactor:` / `test:` / `chore:` prefix and a short lowercase subject, then a plain-prose body wrapped at ~72 columns that says what was wrong or needed and why the change is right: the concrete symptom ("`32561` parsed as 61"), the cause, what changed, how it was checked (tests, a live run, numbers). Name the actual files, functions and figures. No marketing words, no bullet lists of every file touched, no "This commit ...". A one-line change can have a one-line message.

## What this is

FDS (CSD355) project: a measurement layer around LLM data-analysis agents. One labelled corpus of agent runs feeds two analyses: a silent-failure classifier (predicts a wrong-but-confident answer from data + trace features) and a fault-injection study of how three agent topologies degrade. The spec is `Shaurya_Pragun_FDS_Midsem_Report.pdf` (untracked); workflow blocks A–G in the code refer to its Figure 1.

Branches: `main` has the whole study, blocks A–G plus the final report builder. `viva-30` is frozen at the ~30% milestone (blocks A+B only) for the midsem viva — never change it.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate && pip install -e ".[model,dev,report]"
make test                                   # python -m pytest tests/ -q
python -m pytest tests/test_labelling.py::test_parse_plain_number -q   # single test
make data questions                         # datasets/raw/ + datasets/question_banks/
make smoke                                  # 2 questions/dataset end to end, MockLLM fallback
python scripts/generate_runs.py --limit 2 --no-mock --out results/live.jsonl
python scripts/label_runs.py --runs results/live.jsonl
AGENTREL_MODEL=llama3.1:8b python scripts/generate_runs.py --canonical-only --no-mock --out results/bench_llama3.1.jsonl   # model benchmark design
make relate figures report                  # block G, figures, Shaurya_Pragun_FDS_Project_Report.pdf
```

`make all` runs the full chain: runs → label → train → faults → label-faults → robustness → relate (~4.5 h on an M3 with qwen3:8b; ~12 s a baseline run, ~17 s a fault run). `make report` then draws `results/figures/` and builds `Shaurya_Pragun_FDS_Project_Report.pdf`; its interpretive prose lives in `report/findings.json` and must be re-checked by hand whenever the numbers change. Linting: `ruff` (dev dependency, no config beyond defaults).

## LLM backend

Local Ollama only (`configs/experiment.yaml` → `base_llm`, override model with `AGENTREL_MODEL`). The base model is `qwen3:8b` with `think: false`, chosen by a 4-model benchmark (tied with llama3.1:8b on accuracy, faster, loops less); leaving thinking on spends ~300 tokens a step. Run one model at a time: two 8B models on a 16 GB machine evict each other on every request and runs slow ~10x. If Ollama is unreachable, `build_llm` silently falls back to `MockLLM`, whose answers are meaningless — pass `--no-mock` whenever numbers are meant to be real.

## Architecture (the parts that span files)

- **Data** (`data/`): `registry.py` defines the 5 datasets; titanic and adult_census download, the other three are seeded synthetic stand-ins. `questions.py` holds 10 templates per dataset, each with a deterministic pandas `truth` function and 4 paraphrases; `expand_questions` emits canonical + 4 = 5 phrasings, so the base corpus is 750 runs, not the report's 600.
- **Harness** (`topology/`, `agents/`, `harness/`): all three topologies share `ACTION_PROTOCOL`, `initial_history()` and `parse_action()` in `topology/base.py`, the same tool registry (`agents/tools.py`) and the same model — topology is the *only* manipulated variable, so keep shared prompt/parse changes in `base.py`. `parse_action` deliberately tolerates the model's protocol deviations; a malformed reply becomes a final answer (a likely failure), never a dropped run. Every tool call is a `ToolCall` in a `RunTrace` (one JSONL line per run).
- **Labelling** (`labelling/`): `parse.py` extracts a number/category from free text; `label.py` applies Eq. 1 (1% relative tolerance numeric, exact match categorical; label 1 = failure) plus paraphrase consistency; `clean.py` keeps timed-out runs (flagged) and IQR-caps heavy-tailed counts with `_capped` audit columns. Bugs here mislabel the whole corpus silently — add a regression test for any parser change.
- **Analysis**: `features/` (names in `features/schema.py` are the contract; each block asserts it produced exactly those keys), `model/` (class-weighted LightGBM + baselines), `faults/` (injection + Delta_{T,f} with bootstrap CIs), `relate.py` (block G: out-of-fold predictability, per-topology drivers, and the baseline-trained predictor scored on fault runs).

## Methodology invariants (break these and every number is wrong)

- Topology is never a classifier feature — it is what the robustness study manipulates.
- Splits group on `template_id` (paraphrases are near duplicates); the bootstrap resamples templates, not runs.
- The F2 threshold is chosen on validation and frozen for test.
- Labelling a non-baseline corpus writes `<stem>_labelled.parquet`; never let it overwrite `results/labelled.parquet`.

## Gotchas

- `schema_digest()` is deliberately qualitative (no row counts, missing rates or distinct counts): a number shown in the preamble is an answer the agent never has to compute. Runs generated before commit 7e5b5b8 used a leaky preamble; exclude titanic.q02, adult.q01, nyc.q02, nyc.q05, olist.q01 and miss.q01 from anything built on them.
- The 1.5*IQR templates (titanic.q08, adult.q09, nyc.q10) need `aggregate` with `op: quantile`; corpora from before it existed fail them for every model.
- Changing the prompt, the tools or the model changes what is being measured: regenerate both corpora together, never mix runs across such a change.
