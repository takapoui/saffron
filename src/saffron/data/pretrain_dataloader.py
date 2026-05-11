import logging

import numpy as np
import torch

from .base_dataloader import BaseDataLoader

logger = logging.getLogger(__name__)


class PretrainDataLoader(BaseDataLoader):
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
