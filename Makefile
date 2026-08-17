# OptiVision RAG — common tasks.
# On Windows use `make` from Git Bash, or run the commands directly.

PY ?= .venv/Scripts/python.exe
CONFIG ?= configs/colsmol.yaml
CORPUS ?= data/corpus

.PHONY: help install corpus index search bench bench-synthetic figures test lint app clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-18s %s\n", $$1, $$2}'

install:  ## install the package and all extras into .venv
	$(PY) -m pip install -e ".[vlm,bench,app,dev]"

corpus:  ## generate the synthetic document corpus + ground-truth queries
	$(PY) -m optivision.cli make-corpus $(CORPUS) --docs 30 --pages 2

index:  ## build the index (CONFIG=configs/....yaml)
	$(PY) -m optivision.cli index $(CORPUS)/pdfs -c $(CONFIG)

search:  ## search the index (Q="your query")
	$(PY) -m optivision.cli search "$(Q)" -c $(CONFIG)

bench:  ## full ablation table with the real model
	$(PY) -m optivision.cli bench $(CORPUS)/pdfs $(CORPUS)/queries.json -c $(CONFIG) --out reports/colsmol --sweep --cache data/cache/colsmol.npz

bench-synthetic:  ## plumbing-only ablation, no model download
	$(PY) -m optivision.cli bench $(CORPUS)/pdfs $(CORPUS)/queries.json -c configs/synthetic.yaml --out reports/synthetic

figures:  ## render keep-mask figures for the report
	$(PY) -m optivision.cli explain $(CORPUS)/pdfs -c $(CONFIG) --limit 4 --out reports/figures

test:  ## run the test suite
	$(PY) -m pytest -q

lint:  ## ruff check + format check
	$(PY) -m ruff check src tests && $(PY) -m ruff format --check src tests

app:  ## launch the demo UI
	$(PY) -m streamlit run app/streamlit_app.py -- --config $(CONFIG)

clean:  ## remove generated indexes and reports (keeps the corpus)
	rm -rf data/index reports/bench_indexes reports/colsmol reports/synthetic .pytest_cache
