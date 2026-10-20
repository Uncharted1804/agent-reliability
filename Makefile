.PHONY: install data questions runs label test smoke all clean

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

test:
	python -m pytest tests/ -q

# end-to-end on a tiny slice, offline-safe (MockLLM fallback)
smoke: data questions
	python scripts/generate_runs.py --limit 2 --out results/smoke.jsonl
	python scripts/label_runs.py --runs results/smoke.jsonl

all: install data questions runs label

clean:
	rm -rf results/*.jsonl results/*.parquet results/*.csv datasets/raw
