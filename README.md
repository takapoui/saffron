# saffron

A GPT-2 pretraining framework built from scratch in PyTorch — supporting DDP multi-GPU training, `torch.compile`, MFU tracking, and wandb logging.

## Experiments & Results

### Pretraining

| Experiment | Model | Dataset | Hellaswag | Val Loss | MFU | Hardware |
|---|---|---|---|---|---|---|
| [001 — Baseline](experiments/001_gpt2_small_baseline.md) | GPT-2 small (124M) | fineweb-edu 10B | 30.4% | 3.07 | 33% | 8x A100 40GB |

Full details and training curves in [`experiments/`](experiments/).

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

**Pretraining:**
```bash
PYTHONPATH=src .venv/bin/python scripts/prep_pretrain_data.py --config configs/gpt2_small.json
```

**SFT:**
```bash
PYTHONPATH=src .venv/bin/python scripts/prep_sft_data.py --config configs/gsm8k_train.json
PYTHONPATH=src .venv/bin/python scripts/prep_sft_data.py --config configs/gsm8k_test.json
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
