# saffron

A PyTorch codebase for studying small-scale language models end-to-end: pretraining, fine-tuning, and post-training. The goal is to take a model far enough that it does something tangible, like simple math, while keeping every stage of the stack visible and modifiable: architecture (GPT-2 from scratch, HuggingFace adapters for larger models), training loop (DDP, `torch.compile`, mixed-precision, MFU tracking, wandb logging), evaluation (loss, Hellaswag, GSM8K, generation samples), data pipelines, teacher distillation (vLLM with rejection sampling), and RL post-training (GRPO with reward design). Minimal and explicit by design; external dependencies are used when they're not the part being studied.

The experiments below track the project's progress. Built as a personal learning project, where each experiment is a way to understand a piece of the stack in depth. The codebase is kept readable enough that someone else could follow along. Cluster experiments run on [Lambda Labs](https://lambda.ai/) GPUs.

## Experiments & Results

### Pretraining

| Experiment | Model | Dataset | Hellaswag | Val Loss | MFU | Hardware |
|---|---|---|---|---|---|---|
| [001 — Baseline](experiments/001_gpt2_small_baseline.md) | GPT-2 small (124M) | fineweb-edu 10B | 30.4% | 3.07 | 33% | 8x A100 SXM4 40GB |
| [006 — NeoGPT architecture](experiments/006_neogpt_architecture.md) | NeoGPT (114M) | fineweb-edu 10B | 30.9% | 3.05 | 31% | 8x A100 SXM4 40GB |

### SFT

| Experiment | Model | Dataset | GSM8K | MFU | Hardware |
|---|---|---|---|---|---|
| [002 — SFT on GSM8K](experiments/002_sft_qwen_gsm8k.md) | Qwen2.5-0.5B base (500M) | GSM8K | 35.9% | 36% | 1x A10 PCIe 24GB |
| [003 — SFT on GSM8K with teacher distillation](experiments/003_sft_qwen_gsm8k_teacher_distilled.md) | Qwen2.5-0.5B base (500M) | GSM8K (Qwen2.5-Math-7B teacher) | 47.7% | 37% | 1x A100 SXM4 40GB |
| [^ Same](experiments/003_sft_qwen_gsm8k_teacher_distilled.md) | Qwen2.5-1.5B base (1.5B) | GSM8K (Qwen2.5-Math-7B teacher) | **72.8%** | 51% | 1x A100 SXM4 40GB |

### RL

| Experiment | Model | Task | Accuracy | Hardware |
|---|---|---|---|---|
| [004 — GRPO on Countdown](experiments/004_rl_grpo_qwen_1.5b_countdown.md) | Qwen2.5-1.5B base | Countdown 3-num | 47.66% | 1x A100 SXM4 40GB |
| [005 — Reasoning emergence on Countdown](experiments/005_rl_grpo_qwen_3b_countdown.md) | Qwen2.5-3B base | Countdown 3+4-num | 38.28% | 1x GH200 96GB |

Full details and training curves in [`experiments/`](experiments/).

## Setup

```bash
git clone https://github.com/takapoui/saffron.git
cd saffron
```

**Local:**
```bash
make install-local  # creates .venv and installs all dependencies
make kernel         # registers the Jupyter kernel
```

**Cluster (e.g. LambdaLabs):**
```bash
make setup-linux      # installs Python and system deps
make install-cluster  # uses system torch (Lambda Stack) + installs remaining deps via uv
make install-vllm     # additionally installs vllm (only needed for teacher sampling)
```

## Data Preparation

**Pretraining:**
```bash
.venv/bin/python scripts/prep_pretrain_data.py --config configs/pretraining/gpt2_small.json
```

Each config file contains a `prep` section with dataset and tokenization settings alongside the training config.

**SFT:**
```bash
.venv/bin/python scripts/prep_sft_data.py --config configs/sft/qwen_0.5b_gsm8k.json --key prep_train
.venv/bin/python scripts/prep_sft_data.py --config configs/sft/qwen_0.5b_gsm8k.json --key prep_val
```

**Teacher sampling (GSM8K):**
```bash
# Step 1 — generate teacher answers with round-robin validation
.venv/bin/python scripts/sample_teacher.py --config configs/distillation/teacher_gsm8k.json --key Qwen2.5-Math-7B-Instruct

# Step 2 — tokenize teacher answers and val split into npy shards
.venv/bin/python scripts/prep_sft_data.py --config configs/distillation/teacher_gsm8k.json --key prep_train
.venv/bin/python scripts/prep_sft_data.py --config configs/distillation/teacher_gsm8k.json --key prep_val
```

**RL prompts (Countdown):**
```bash
.venv/bin/python scripts/prep_rl_data.py --config configs/rl/countdown.json
```

## Training

**Pretraining:**
```bash
# single device (CPU, MPS, or CUDA)
.venv/bin/python scripts/run_train.py --config configs/pretraining/gpt2_small.json

# multi GPU (e.g. 8 GPUs)
.venv/bin/python -m torch.distributed.run --nproc_per_node=8 scripts/run_train.py --config configs/pretraining/gpt2_small.json
```

**SFT:**
```bash
.venv/bin/python scripts/run_train.py --config configs/sft/qwen_0.5b_gsm8k.json
```

**RL (GRPO):**
```bash
.venv/bin/python scripts/run_rl_train.py --config configs/rl/countdown.json
```

## Development

```bash
make lint         # run ruff and pyright
make test         # run pytest
```

## Credits

[nanoGPT](https://github.com/karpathy/nanoGPT) / [nanochat](https://github.com/karpathy/nanochat): Karpathy's minimalism philosophy and self-contained GPT-2 implementation shaped saffron's style. The peak-FLOPs tables are borrowed from nanochat.

[nano-aha-moment](https://github.com/McGill-NLP/nano-aha-moment) (McGill NLP): the GRPO training setup and Countdown task configuration in experiments 004 and 005 are directly inspired by their work.

Claude (Anthropic): helped with cumbersome work like writing tests, boilerplate, and some refactors.
