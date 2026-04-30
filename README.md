# saffron

LLM experiments in Python.

## Setup

```bash
make setup-linux  # Ubuntu/Debian (e.g. LambdaLabs) — installs Python and system deps
make install      # creates .venv and installs all dependencies
make kernel       # registers the Jupyter kernel
```

## Usage

```bash
PYTHONPATH=src .venv/bin/python scripts/prep_data.py \
    --dataset HuggingFaceFW/fineweb-edu \
    --name sample-10BT \
    --shard_size 100000000
```

## Development

```bash
make lint         # run ruff and pyright
make test         # run pytest
```
