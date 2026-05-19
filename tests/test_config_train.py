"""Tests for TrainConfig.from_dict() — nested eval-block parsing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from saffron.config import TrainConfig
from saffron.eval import EvalGenerateConfig, EvalGSM8KConfig, EvalHellaswagConfig, EvalLossConfig


def test_train_config_parses_nested_eval_blocks(tmp_path: Path) -> None:
    d: dict[str, Any] = {
        "max_steps": 200,
        "optimizer": {"lr": 3e-4, "weight_decay": 0.1},
        "schedule": {"warmup_steps": 20, "min_lr_ratio": 0.1},
        "grad_clip": 1.0,
        "total_batch_size": 1024,
        "compile_model": False,
        "eval_loss": {"every": 50, "steps": 10},
        "eval_generate": {
            "every": 100,
            "prompt": "Once upon a time",
            "samples": 3,
            "max_tokens": 64,
            "use_chat_template": True,
            "temperature": 0.8,
            "top_k": 40,
        },
        "eval_hellaswag": {"every": None},
        "eval_gsm8k": {"every": None, "max_tokens": 300, "gen_batch_size": 32},
        "checkpoint_dir": str(tmp_path / "ckpt"),
        "checkpoint_every": 100,
        "resume_from": None,
        "log_every": 10,
        "wandb_project": None,
    }

    cfg = TrainConfig.from_dict(d)

    assert isinstance(cfg.eval_loss, EvalLossConfig)
    assert cfg.eval_loss.every == 50
    assert cfg.eval_loss.steps == 10

    assert isinstance(cfg.eval_generate, EvalGenerateConfig)
    assert cfg.eval_generate.every == 100
    assert cfg.eval_generate.prompt == "Once upon a time"
    assert cfg.eval_generate.use_chat_template is True
    assert cfg.eval_generate.temperature == pytest.approx(0.8)  # type: ignore[reportUnknownMemberType]
    assert cfg.eval_generate.top_k == 40

    assert isinstance(cfg.eval_hellaswag, EvalHellaswagConfig)
    assert cfg.eval_hellaswag.every is None

    assert isinstance(cfg.eval_gsm8k, EvalGSM8KConfig)
    assert cfg.eval_gsm8k.every is None
    assert cfg.eval_gsm8k.max_tokens == 300
    assert cfg.eval_gsm8k.gen_batch_size == 32

    # resume_weights_only defaults to False when absent from dict
    assert cfg.resume_weights_only is False
