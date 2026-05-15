from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset  # type: ignore[reportUnknownVariableType]

from ..tokenizer import Tokenizer
from .config import RLPrepConfig

logger = logging.getLogger(__name__)


def prepare_rl_dataset(prep_config: RLPrepConfig) -> None:

    enc = Tokenizer.from_name(prep_config.tokenizer)
    ds: Any = load_dataset(
        prep_config.dataset,
        prep_config.name if prep_config.name else None,
        split=prep_config.dataset_split,
    )

    def preprocess(example: dict[str, Any]) -> dict[str, Any]:
        user_message = prep_config.prompt_template.format(
            nums=example["nums"], target=example["target"]
        )
        messages = [
            {"role": "system", "content": prep_config.system_prompt},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": prep_config.assistant_prefill},
        ]
        input_ids = enc.apply_chat_template(
            messages,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        return {"input_ids": input_ids}

    nprocs = max(1, (os.cpu_count() or 2) // 2)
    ds = ds.map(preprocess, num_proc=nprocs, desc="Tokenizing prompts")

    n_total = len(ds)
    n_val = prep_config.val_size
    if n_val >= n_total:
        raise ValueError(f"val_size ({n_val}) must be smaller than dataset ({n_total}).")
    n_train = n_total - n_val
    train_ds = ds.select(range(n_train))
    val_ds = ds.select(range(n_train, n_total))

    data_root = prep_config.data_root
    data_root.mkdir(parents=True, exist_ok=True)
    with open(data_root / "meta.json", "w") as f:
        json.dump({"tokenizer": prep_config.tokenizer}, f)

    _write_jsonl(train_ds, data_root / "train.jsonl")
    _write_jsonl(val_ds, data_root / "val.jsonl")

    _log_token_stats(train_ds, "train")
    _log_token_stats(val_ds, "val")


def _write_jsonl(ds: Any, path: Path) -> None:
    n = 0
    with open(path, "w") as f:
        for example in ds:
            record = {
                "nums": example["nums"],
                "target": example["target"],
                "input_ids": example["input_ids"],
            }
            f.write(json.dumps(record) + "\n")
            n += 1
    logger.info(f"Wrote {n} examples to {path}")


def _log_token_stats(ds: Any, split: str) -> None:
    lengths = np.array([len(ex["input_ids"]) for ex in ds])
    logger.info(
        f"{split} token lengths over {len(lengths)} examples — "
        f"avg: {lengths.mean():.0f} | max: {lengths.max()} | "
        f"p50: {np.percentile(lengths, 50):.0f} | p99: {np.percentile(lengths, 99):.0f}"
    )
