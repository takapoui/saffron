.PHONY: help setup-linux venv install install-local install-cluster install-vllm activate-venv kernel lint lint-fix test clean

VENV := .venv
PYTHON_VERSION := $(shell grep 'requires-python' pyproject.toml | sed 's/[^0-9]*\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/')
SYSTEM_PYTHON_VERSION := $(shell python3 --version 2>&1 | sed 's/Python \([0-9]*\.[0-9]*\).*/\1/')
PYTHON := $(VENV)/bin/python
UV := $(VENV)/bin/uv
RUFF := $(PYTHON) -m ruff
PYRIGHT := $(PYTHON) -m pyright
PYTEST := $(PYTHON) -m pytest
NODE_PATH_PREFIX := /opt/homebrew/bin:$(PATH)
help:
	@echo "Available targets:"
	@echo "  make setup-linux          - Install system dependencies on Ubuntu/Debian (e.g. LambdaLabs)"
	@echo "  make venv                 - Create the local .venv if it does not exist"
	@echo "  make install-local        - Install all dependencies for local development"
	@echo "  make install-cluster      - Install cluster dependencies using system torch (Lambda Labs)"
	@echo "  make install-vllm         - Install vllm on top of cluster dependencies (for teacher sampling)"
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
	$(VENV)/bin/pip install uv

$(PYTHON)-system:
	python$(SYSTEM_PYTHON_VERSION) -m venv --system-site-packages $(VENV)
	$(VENV)/bin/pip install uv

venv: $(PYTHON)

install-local: $(PYTHON)
	$(UV) pip install -r requirements.txt
	$(UV) pip install -e .
	$(PYTHON) -m pre_commit install

install-cluster: $(PYTHON)-system
	$(UV) pip install -r requirements-cluster.txt
	$(UV) pip install -e .
	$(PYTHON) -c "import flash_attn" 2>/dev/null || $(UV) pip install flash-attn --no-build-isolation --no-deps || true

install-vllm: install-cluster
	$(UV) pip install vllm

activate-venv: venv
	@echo "Run this in your shell:"
	@echo "source $(VENV)/bin/activate"

kernel: install-local
	$(PYTHON) -m ipykernel install --sys-prefix --name saffron-venv --display-name "Python (.venv)"

lint:
	$(RUFF) check .
	$(RUFF) format --check .
	PATH="$(NODE_PATH_PREFIX)" $(PYRIGHT)

lint-fix:
	$(RUFF) format .
	$(RUFF) check . --fix

test:
	$(PYTEST) tests/ -v

clean:
	rm -rf $(VENV)
