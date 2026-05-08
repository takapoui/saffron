# saffron

LLM experiments in Python.

## Setup

**Local:**
```bash
make install-local  # creates .venv and installs all dependencies
make kernel         # registers the Jupyter kernel
```

**Cluster (e.g. LambdaLabs):**
```bash
make setup-linux      # installs Python and system deps
make install-cluster  # uses system torch (Lambda Stack) + installs remaining deps via uv
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
PYTHONPATH=src .venv/bin/torchrun --nproc_per_node=8 scripts/run_train.py --config configs/gpt2_small.json
```

## Development

```bash
make lint         # run ruff and pyright
make test         # run pytest
```
