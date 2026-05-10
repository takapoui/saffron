import json
import logging

import numpy as np
import torch

from ..config import RunConfig
from .config import DataConfig

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(
        self,
        data_config: DataConfig,
        run_config: RunConfig,
        split: str,
    ) -> None:
        # We assume data_root has files called train_xxx.npy and val_yyy.py
        # The files are already tokenized
        self.B = data_config.batch_size
        self.T = data_config.seq_len
        self.rank = run_config.ddp_rank

        meta_path = data_config.data_root / "meta.json"
        assert meta_path.exists(), (
            f"No meta.json found in {data_config.data_root}. Re-run data prep to generate it."
        )
        with open(meta_path) as f:
            meta = json.load(f)
        self.tokenizer_name: str = meta["tokenizer"]
        if self.tokenizer_name != data_config.tokenizer:
            raise ValueError(
                f"Data in '{data_config.data_root}' was built with tokenizer "
                f"'{self.tokenizer_name}' but config specifies '{data_config.tokenizer}'. "
                "Re-run data prep with the correct tokenizer."
            )
        self.world_size = run_config.ddp_world_size

        self.shards = sorted(
            [str(p) for p in data_config.data_root.iterdir() if p.name.startswith(split)]
        )
        assert len(self.shards) > 0, f"No shards found in {data_config.data_root} for split {split}"
        if self.rank == 0:
            logger.info(
                f"Found {len(self.shards)} shards for split {split} in {data_config.data_root}"
            )

        self.reset()

    def reset(self) -> None:
        self.current_shard = 0
        self.tokens = self._load_shard(self.current_shard)
        self.current_position = self.B * self.T * self.rank

    def advance(self, tokens: int) -> None:
        while tokens > 0:
            remaining_in_shard = len(self.tokens) - self.current_position
            if tokens < remaining_in_shard:
                self.current_position += tokens
                tokens = 0
            else:
                tokens -= remaining_in_shard
                self.current_shard = (self.current_shard + 1) % len(self.shards)
                self.tokens = self._load_shard(self.current_shard)
                self.current_position = self.B * self.T * self.rank

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        buf = self.tokens[self.current_position : self.current_position + self.B * self.T + 1]
        x, y = buf[:-1].reshape(self.B, self.T), buf[1:].reshape(self.B, self.T)
        self.current_position += self.B * self.T * self.world_size

        if self.current_position + self.B * self.T + 1 >= len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self._load_shard(self.current_shard)
            self.current_position = self.B * self.T * self.rank
        return x, y

    def _load_shard(self, shard_idx: int) -> torch.Tensor:
        assert shard_idx < len(self.shards)
        fn = self.shards[shard_idx]
        npt = np.load(fn)
        if self.B * self.T * self.world_size + 1 >= len(npt):
            logger.warning(
                f"Shard {shard_idx} is too small. If this warning persists across shards, "
                "it means you need to make shards bigger."
            )

        return torch.tensor(npt, dtype=torch.long)
