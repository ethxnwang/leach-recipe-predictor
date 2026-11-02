PYTHON ?= python3.11
VENV := .venv
BIN := $(VENV)/bin

.PHONY: setup lint format test simulate lab-data eda train evaluate recommend study pipeline serve app clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev,app]"

lint:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

test:
	$(BIN)/pytest

simulate:
	$(BIN)/leach simulate

lab-data:
	$(BIN)/leach simulate --n-trials 40 --n-batches 4 --seed 7 --prefix LAB --out data/example_lab_trials.csv

eda:
	$(BIN)/leach eda

train:
	$(BIN)/leach train

evaluate:
	$(BIN)/leach evaluate data/example_lab_trials.csv

recommend:
	$(BIN)/leach recommend --feed-type B --grade 2.5 --particle-size 120

study:
	$(BIN)/leach trial-study

pipeline: simulate lab-data eda train evaluate study

serve:
	$(BIN)/uvicorn leach.api:app --reload --port 8000

app:
	$(BIN)/streamlit run app/streamlit_app.py

clean:
	rm -rf models/*.joblib .pytest_cache .ruff_cache
