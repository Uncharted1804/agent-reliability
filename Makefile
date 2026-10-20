.PHONY: install data questions runs label train faults label-faults robustness test smoke all clean

install:
	pip install -e .

data:
	python scripts/download_datasets.py

questions:
	python scripts/build_question_banks.py

runs:
	python scripts/generate_runs.py

label:
	python scripts/label_runs.py --runs results/runs.jsonl

train:
	python scripts/train_classifier.py --runs results/runs.jsonl

faults:
	python scripts/generate_fault_runs.py --no-mock

label-faults:
	python scripts/label_runs.py --runs results/runs_faults.jsonl

robustness:
	python scripts/robustness.py

test:
	python -m pytest tests/ -q

# end-to-end on a tiny slice, offline-safe (MockLLM fallback)
smoke: data questions
	python scripts/generate_runs.py --limit 2 --out results/smoke.jsonl
	python scripts/label_runs.py --runs results/smoke.jsonl

# the full study: corpus -> labels -> classifier -> faults -> robustness
all: install data questions runs label train faults label-faults robustness

clean:
	rm -rf results/*.jsonl results/*.parquet results/*.csv results/*.json datasets/raw
