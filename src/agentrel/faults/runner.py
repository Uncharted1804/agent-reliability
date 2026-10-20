"""Fault-injected run generator (block E).

Re-runs questions from the question banks under each fault, once per topology,
recording the same RunTrace schema as the baseline corpus plus the fault
provenance. The output is a separate corpus (runs_faults.jsonl) so the
classifier corpus stays clean: the predictor is trained on normal operation,
while the robustness study compares faulted against baseline.

Scale. The report's full design is 600 x 3 fault-injected runs, but on a local
CPU-served model a run costs ~15s, so the full set is ~10 hours of compute. The
number of phrasings per template is therefore a config knob
(`fault_runs.paraphrases_per_template`, default 1 = the canonical phrasing
only), which keeps every template and every topology represented -- the two
things Delta_{T,f} is estimated across -- while cutting the cost to a third.
Raise it to 5 to reproduce the full design.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..agents.llm import build_llm
from ..agents.tools import Session
from ..data.registry import load_tables
from ..harness.trace import RunTrace
from ..topology import build_topology
from ..utils.config import REPO_ROOT, Config
from ..utils.io import append_jsonl
from .inject import FAULTS, FaultInjector

QUESTION_BANK_DIR = REPO_ROOT / "datasets" / "question_banks"


def _load_bank(dataset: str) -> dict:
    path = QUESTION_BANK_DIR / f"{dataset}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"question bank for {dataset} missing — run scripts/build_question_banks.py")
    return json.loads(path.read_text())


def _selected_questions(bank: dict, paraphrases: int, limit: int | None) -> list[dict]:
    """Questions to fault: the first `paraphrases` phrasings of every template."""
    questions = [q for q in bank["questions"] if q["paraphrase_index"] < paraphrases]
    return questions[:limit] if limit else questions


def generate_fault_runs(cfg: Config, *, out_path: str | Path | None = None,
                        faults: list[str] | None = None, paraphrases: int | None = None,
                        limit: int | None = None, allow_mock: bool = True) -> Path:
    """Run every (question, topology, fault) combination and log the traces."""
    out_path = Path(out_path) if out_path else cfg.log_dir / "runs_faults.jsonl"
    if out_path.exists():
        out_path.unlink()

    faults = faults or cfg.faults or list(FAULTS)
    unknown = sorted(set(faults) - set(FAULTS))
    if unknown:
        raise ValueError(f"unknown faults {unknown}; known: {FAULTS}")
    paraphrases = paraphrases or cfg.fault_paraphrases

    llm = build_llm(cfg.llm, allow_mock=allow_mock)

    total = 0
    for dataset in cfg.datasets:
        qs = _selected_questions(_load_bank(dataset), paraphrases, limit)
        total += len(qs) * len(cfg.topologies) * len(faults)
    print(f"Generating {total} fault-injected runs with {llm.__class__.__name__} "
          f"({cfg.llm.model}); faults={faults} -> {out_path}", flush=True)

    t_start = time.time()
    n_runs = 0
    for dataset in cfg.datasets:
        bank = _load_bank(dataset)
        tables = load_tables(dataset)
        for q in _selected_questions(bank, paraphrases, limit):
            for topo_name in cfg.topologies:
                for fault in faults:
                    run_id = f"{q['question_id']}::{topo_name}::{fault}"
                    trace = RunTrace(
                        run_id=run_id, dataset=dataset, template_id=q["template_id"],
                        question_id=q["question_id"],
                        paraphrase_index=q["paraphrase_index"], topology=topo_name,
                        model=cfg.llm.model, question_text=q["text"],
                    )
                    # the injector is seeded from the run id, so this run is
                    # faulted identically if the corpus is regenerated
                    injector = FaultInjector(fault=fault, run_id=run_id)
                    session = Session(tables={k: v.copy() for k, v in tables.items()})
                    topo = build_topology(topo_name, llm, max_steps=cfg.max_agent_steps,
                                          injector=injector)
                    try:
                        topo.run(trace, session)
                    except Exception as exc:  # noqa: BLE001 — harness-level failure
                        trace.harness_error = f"{exc.__class__.__name__}: {exc}"
                        trace.finish(None)
                    for key, value in injector.describe().items():
                        setattr(trace, key, value)
                    append_jsonl(out_path, trace.to_dict())
                    n_runs += 1
                    if n_runs % 10 == 0 or n_runs == total:
                        elapsed = time.time() - t_start
                        rate = elapsed / n_runs
                        print(f"  [{n_runs}/{total}] {dataset} {topo_name} {fault} "
                              f"{rate:.1f}s/run  eta {rate*(total-n_runs)/60:.0f}m",
                              flush=True)

    print(f"Generated {n_runs} fault-injected runs -> {out_path}")
    return out_path
