# saffron

LLM experiments in Python.

## Setup

```bash
make setup-linux  # Ubuntu/Debian (e.g. LambdaLabs) — installs Python and system deps
make install      # creates .venv and installs all dependencies
make kernel       # registers the Jupyter kernel (local only)
```

On a GPU cluster (e.g. LambdaLabs), reuse the system-installed torch to skip the heavy download:

```bash
SYSTEM_SITE_PACKAGES=1 make install
```

## Data Preparation

```bash
PYTHONPATH=src .venv/bin/python scripts/prep_data.py --config configs/gpt2_small.json
```

## Training

```bash
# single device (CPU, MPS, or CUDA)
PYTHONPATH=src .venv/bin/python scripts/run_train.py --config configs/gpt2_small.json

# multi GPU (e.g. 8 GPUs)
PYTHONPATH=src torchrun --nproc_per_node=8 scripts/run_train.py --config configs/gpt2_small.json
```

## Development

```bash
make lint         # run ruff and pyright
make test         # run pytest
```
