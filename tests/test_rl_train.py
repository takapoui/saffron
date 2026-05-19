"""Tests for RLTrainer checkpoint save/resume.

We mock _step so tests don't need to run real rollouts (which require a full
HF model + tokenizer). The harness — save timing, resume start, RNG restoration —
is what we care about here.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest
import torch

from saffron.config import OptimizerConfig, RLConfig, RunConfig
from saffron.data import RLDataLoader
from saffron.model import GPT2, GPT2Config
from saffron.train import RLTrainer


def _run_config() -> RunConfig:
    return RunConfig(
        device="cpu",
        device_type="cpu",
        use_ddp=False,
        ddp_rank=0,
        ddp_local_rank=0,
        ddp_world_size=1,
    )


def _rl_config(
    tmp_path: Path,
    *,
    num_steps: int = 4,
    checkpoint_every: int = 2,
    resume_from: Path | None = None,
) -> RLConfig:
    return RLConfig(
        num_steps=num_steps,
        grad_clip=1.0,
        optimizer=OptimizerConfig(lr=1e-4, weight_decay=0.0),
        n_prompts_per_batch=1,
        group_size=1,
        max_new_tokens=4,
        temperature=1.0,
        clip_eps=0.2,
        kl_coef=0.0,
        microbatch_size=1,
        eval_every=None,
        eval_n_prompts=1,
        checkpoint_dir=tmp_path / "ckpt",
        checkpoint_every=checkpoint_every,
        resume_from=resume_from,
        resume_weights_only=False,
        log_every=10_000,
        wandb_project=None,
    )


def _make_model() -> GPT2:
    cfg = GPT2Config(vocab_size=50, n_embd=8, block_size=4, n_layer=1, n_head=1)
    m = GPT2(cfg)
    m.supports_compile = lambda device_type: False  # type: ignore[method-assign]
    return m


@pytest.fixture
def tiny_model() -> GPT2:
    torch.manual_seed(0)  # type: ignore[reportUnknownMemberType]
    return _make_model()


@pytest.fixture
def tiny_ref_model() -> GPT2:
    torch.manual_seed(0)  # type: ignore[reportUnknownMemberType]
    return _make_model()


class _FakeRLLoader:
    """Minimal stub — only the tokenizer-name check touches it when _step is mocked."""

    def __init__(self, tokenizer_name: str = "gpt2") -> None:
        self.tokenizer = MagicMock()
        self.tokenizer.name = tokenizer_name


def _make_trainer(
    model: GPT2,
    ref_model: GPT2,
    cfg: RLConfig,
) -> RLTrainer:
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.lr)
    return RLTrainer(
        model=model,
        ref_model=ref_model,
        optimizer=optimizer,
        train_loader=cast(RLDataLoader, _FakeRLLoader()),
        val_loader=cast(RLDataLoader, _FakeRLLoader()),
        rl_config=cfg,
        run_config=_run_config(),
    )


def test_save_writes_file_with_expected_keys(
    tmp_path: Path, tiny_model: GPT2, tiny_ref_model: GPT2
) -> None:
    cfg = _rl_config(tmp_path)
    trainer = _make_trainer(tiny_model, tiny_ref_model, cfg)
    trainer._save_checkpoint(step=5)  # pyright: ignore[reportPrivateUsage]

    files = sorted(cfg.checkpoint_dir.glob("ckpt_*.pt"))
    assert len(files) == 1
    assert files[0].name == "ckpt_000005.pt"

    payload = torch.load(files[0], weights_only=False)
    assert payload["step"] == 5
    assert "model_dict" in payload
    assert "optimizer_dict" in payload
    assert "model_config" in payload
    assert "rl_config" in payload
    assert "rng_state" in payload


def test_resume_does_not_skip_or_repeat_steps(
    tmp_path: Path, tiny_model: GPT2, tiny_ref_model: GPT2
) -> None:
    """The off-by-one bug: checkpoint saved before _step left step N un-executed on resume.

    Run 4 steps, checkpoint at step 2 (post-step under the fix), then resume.
    The two sessions combined should execute steps 0..3 exactly once each.
    """
    cfg = _rl_config(tmp_path, num_steps=4, checkpoint_every=2)
    trainer = _make_trainer(tiny_model, tiny_ref_model, cfg)

    executed_first: list[int] = []

    def _stub_step_first() -> dict[str, float]:
        executed_first.append(trainer.step)
        return {}

    trainer._step = _stub_step_first  # type: ignore[method-assign]
    trainer.train()
    assert executed_first == [0, 1, 2, 3]

    # Pick the latest periodic checkpoint (step 2; the final-save fallback won't fire here
    # because (num_steps-1)=3 is not a multiple of checkpoint_every=2 — see below).
    ckpts = sorted(cfg.checkpoint_dir.glob("ckpt_*.pt"))
    # We expect checkpoints at steps 0, 2 (periodic) plus final-save at 3.
    assert [p.name for p in ckpts] == ["ckpt_000000.pt", "ckpt_000002.pt", "ckpt_000003.pt"]

    # Resume from the step-2 checkpoint. Steps 3 should be the only one re-executed.
    resume_cfg = _rl_config(
        tmp_path,
        num_steps=4,
        checkpoint_every=2,
        resume_from=cfg.checkpoint_dir / "ckpt_000002.pt",
    )
    resumed = _make_trainer(_make_model(), _make_model(), resume_cfg)

    executed_resumed: list[int] = []

    def _stub_step_resumed() -> dict[str, float]:
        executed_resumed.append(resumed.step)
        return {}

    resumed._step = _stub_step_resumed  # type: ignore[method-assign]
    resumed.train()
    assert executed_resumed == [3]


def test_resume_restores_rng_state(tmp_path: Path, tiny_model: GPT2, tiny_ref_model: GPT2) -> None:
    """Resuming from a checkpoint must continue the RNG sequence, not restart it."""
    cfg = _rl_config(tmp_path, num_steps=4, checkpoint_every=2)

    # Continuous baseline: run all 4 steps, record what _rng produced at each step.
    baseline = _make_trainer(_make_model(), _make_model(), cfg)
    drawn_continuous: list[int] = []

    def _stub_baseline() -> dict[str, float]:
        drawn_continuous.append(int(baseline._rng.integers(1_000_000)))  # pyright: ignore[reportPrivateUsage]
        return {}

    baseline._step = _stub_baseline  # type: ignore[method-assign]
    baseline.train()

    # Resume from ckpt_000002.pt (saved after step 2). The next draw the resumed
    # trainer makes is the step-3 draw — which must equal drawn_continuous[3].
    resume_cfg = _rl_config(
        tmp_path,
        num_steps=4,
        checkpoint_every=2,
        resume_from=cfg.checkpoint_dir / "ckpt_000002.pt",
    )
    resumed = _make_trainer(_make_model(), _make_model(), resume_cfg)
    drawn_resumed: list[int] = []

    def _stub_resumed() -> dict[str, float]:
        drawn_resumed.append(int(resumed._rng.integers(1_000_000)))  # pyright: ignore[reportPrivateUsage]
        return {}

    resumed._step = _stub_resumed  # type: ignore[method-assign]
    resumed.train()

    # Resumed ran only step 3; its draw must match the continuous run's step-3 draw.
    assert drawn_resumed == [drawn_continuous[3]]
