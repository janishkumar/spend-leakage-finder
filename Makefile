# PYTHONPATH is set explicitly: macOS can flag uv's editable-install .pth as hidden,
# which Python then skips. Being explicit works everywhere.
export PYTHONPATH := src:.

.PHONY: setup lint test run run-offline report app all docker

setup:
	uv sync

lint:
	uv run ruff check src app tests

test:
	uv run pytest --cov=leakage --cov-report=term-missing

run:            ## download the latest snapshot (if changed) and run every detector
	uv run python -m leakage.cli run

run-offline:    ## re-run on the newest local snapshot
	uv run python -m leakage.cli run --offline

report:         ## docs/findings.md + reports/review_pack.xlsx
	uv run python -m leakage.cli report

app:
	uv run streamlit run app/streamlit_app.py

all: run report

docker:
	docker build -t spend-leakage-finder .
	docker run --rm -p 8501:8501 -v "$$PWD/data:/app/data" spend-leakage-finder
