.PHONY: help setup-linux venv install activate-venv kernel lint lint-fix test clean

VENV := .venv
PYTHON_VERSION := $(shell grep 'requires-python' pyproject.toml | sed 's/[^0-9]*\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/')
PYTHON := $(VENV)/bin/python
PIP := $(PYTHON) -m pip
RUFF := $(PYTHON) -m ruff
PYRIGHT := $(PYTHON) -m pyright
PYTEST := $(PYTHON) -m pytest
NODE_PATH_PREFIX := /opt/homebrew/bin:$(PATH)
help:
	@echo "Available targets:"
	@echo "  make setup-linux          - Install system dependencies on Ubuntu/Debian (e.g. LambdaLabs)"
	@echo "  make venv                 - Create the local .venv if it does not exist"
	@echo "  make install              - Install minimal requirements and the project into .venv"
	@echo "  make activate-venv        - Print the command to activate .venv"
	@echo "  make kernel               - Register/update the Jupyter kernel from .venv"
	@echo "  make lint                 - Run code linters and formatters from .venv"
	@echo "  make lint-fix             - Auto-fix lint and format issues"
	@echo "  make test                 - Run tests from .venv"
	@echo "  make clean                - Remove .venv"

setup-linux:
	sudo add-apt-repository ppa:deadsnakes/ppa -y
	sudo apt update
	sudo apt install -y python$(PYTHON_VERSION) python$(PYTHON_VERSION)-venv python$(PYTHON_VERSION)-dev
	git config --global core.editor vim

$(PYTHON):
	python$(PYTHON_VERSION) -m venv $(VENV)

venv: $(PYTHON)

install: $(PYTHON)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	$(PYTHON) -m pre_commit install

activate-venv: venv
	@echo "Run this in your shell:"
	@echo "source $(VENV)/bin/activate"

kernel: install
	$(PYTHON) -m ipykernel install --sys-prefix --name saffron-venv --display-name "Python (.venv)"

lint:
	$(RUFF) check .
	$(RUFF) format --check .
	PATH="$(NODE_PATH_PREFIX)" $(PYRIGHT)

lint-fix:
	$(RUFF) format .
	$(RUFF) check . --fix

test:
	PYTHONPATH=src $(PYTEST) tests/ -v

clean:
	rm -rf $(VENV)
