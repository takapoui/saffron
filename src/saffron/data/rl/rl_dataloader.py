from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...tokenizer import Tokenizer

logger = logging.getLogger(__name__)


@dataclass
class RLBatch:
    input_ids: torch.Tensor  # (B, T_prompt) — left-padded with pad_token_id
    attention_mask: torch.Tensor  # (B, T_prompt) — 0 on left-pad
    samples: list[dict[str, Any]]  # length B — reward metadata per example

    @classmethod
    def from_examples(cls, examples: list[dict[str, Any]], pad_token_id: int) -> RLBatch:
        B = len(examples)
        T = max(len(ex["input_ids"]) for ex in examples)
        input_ids = torch.full((B, T), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((B, T), dtype=torch.long)
        for i, example in enumerate(examples):
            L = len(example["input_ids"])
            input_ids[i, T - L :] = torch.tensor(example["input_ids"])
            attention_mask[i, T - L :] = 1
        samples = [{"nums": ex["nums"], "target": ex["target"]} for ex in examples]
        return cls(input_ids, attention_mask, samples)


class RLDataLoader:
    # Note that this does not inherit from BaseDataLoader because the data shape is
    # different: variable-length prompts plus heterogeneous metadata, not fixed-shape tensors.

    def __init__(self, path: Path, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self._validate_meta(path.parent)
        with open(path) as f:
            self.examples = [json.loads(line) for line in f]
            logger.info(f"Loaded {len(self.examples)} examples into memory.")

    def _validate_meta(self, data_root: Path) -> None:
        meta_path = data_root / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No meta.json found in {data_root}. Re-run data prep to generate it."
            )
        with open(meta_path) as f:
            meta = json.load(f)
        if meta["tokenizer"] != self.tokenizer.name:
            raise ValueError(
                f"Data in '{data_root}' was built with tokenizer "
                f"'{meta['tokenizer']}' but config specifies '{self.tokenizer.name}'. "
                "Re-run data prep with the correct tokenizer."
            )

    def __len__(self) -> int:
        return len(self.examples)

    def sample_batch(self, n: int, rng: np.random.Generator) -> RLBatch:
        indices = rng.choice(len(self), size=n, replace=False)
        picked = [self.examples[i] for i in indices]
        return RLBatch.from_examples(picked, pad_token_id=self.tokenizer.pad_token_id)
