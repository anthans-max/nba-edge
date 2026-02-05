SHELL := /bin/sh

VENV_DIR := .venv
VENV_BIN := $(VENV_DIR)/bin
PYTHON := $(shell [ -x "$(VENV_BIN)/python" ] && echo "$(VENV_BIN)/python" || command -v python3)
PIP := $(PYTHON) -m pip

.PHONY: help venv install doctor print-python ingest-nba ingest-odds transform transform-odds features train backtest app app-clean pipeline format lint smoke loop run-app kill-app tail-app

help:
	@echo "Available targets:"
	@echo "  venv         Create .venv if missing"
	@echo "  install      Install Python dependencies"
	@echo "  ingest-nba   Ingest NBA game logs"
	@echo "  ingest-odds  Ingest odds spreads"
	@echo "  transform    Build silver tables (games + odds)"
	@echo "  transform-odds Build silver odds spreads"
	@echo "  features     Build features"
	@echo "  train        Train model"
	@echo "  backtest     Run backtests"
	@echo "  app          Run Streamlit app"
	@echo "  app-clean    Kill port 8501 listener then run Streamlit app"
	@echo "  loop         Run dev loop (smoke + app)"
	@echo "  run-app      Launch Streamlit app in background"
	@echo "  kill-app     Stop Streamlit app"
	@echo "  tail-app     Tail Streamlit log"
	@echo "  smoke        Run smoke tests"
	@echo "  pipeline     Run full pipeline"
	@echo "  format       Format code (if tooling exists)"
	@echo "  lint         Lint code (if tooling exists)"
	@echo "  print-python Show resolved Python interpreter"
	@echo "  doctor       Check Python, pip, and requests"

print-python:
	@echo "$(PYTHON)"

doctor:
	@echo "Python: $(PYTHON)"
	@$(PYTHON) --version
	@$(PIP) --version
	@$(PYTHON) -c "import importlib.util; print('requests:', 'ok' if importlib.util.find_spec('requests') else 'missing')"

venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		python3 -m venv $(VENV_DIR); \
		echo "Created $(VENV_DIR)"; \
	else \
		echo "$(VENV_DIR) already exists"; \
	fi

install: venv
	@if [ ! -f requirements.txt ]; then \
		echo "requirements.txt not found"; \
		exit 1; \
	fi
	@$(PYTHON) -m ensurepip --upgrade >/dev/null 2>&1 || true
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt

ingest-nba:
	@$(PYTHON) -m ingest.ingest_nba_gamelogs

ingest-odds:
	@$(PYTHON) -m ingest.ingest_odds_spreads

transform:
	@$(PYTHON) -m transform.build_silver_games
	@$(PYTHON) -m transform.build_silver_odds

transform-odds:
	@$(PYTHON) -m transform.build_silver_odds

features:
	@$(PYTHON) -m features.build_features

train:
	@$(PYTHON) -m train.train_margin_model

backtest:
	@$(PYTHON) -m train.backtest

app:
	@$(PYTHON) -m streamlit run app/app.py

app-clean:
	@lsof -ti :8501 | xargs -r kill -9 || true
	@$(PYTHON) -m streamlit run app/app.py

smoke:
	@$(PYTHON) -m pytest -q -k smoke

loop:
	@scripts/dev_loop.sh

run-app:
	@scripts/run_app.sh

kill-app:
	@scripts/kill_app.sh

tail-app:
	@tail -f .run/streamlit.log

pipeline: ingest-nba ingest-odds transform transform-odds features train backtest

format:
	@if command -v black >/dev/null 2>&1; then \
		black .; \
	elif [ -x "$(VENV_BIN)/black" ]; then \
		$(VENV_BIN)/black .; \
	else \
		echo "Formatter not installed. Install black to enable formatting."; \
	fi

lint:
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check .; \
	elif command -v flake8 >/dev/null 2>&1; then \
		flake8 .; \
	elif [ -x "$(VENV_BIN)/ruff" ]; then \
		$(VENV_BIN)/ruff check .; \
	elif [ -x "$(VENV_BIN)/flake8" ]; then \
		$(VENV_BIN)/flake8 .; \
	else \
		echo "Linter not installed. Install ruff or flake8 to enable linting."; \
	fi
