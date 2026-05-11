import logging

import numpy as np
import torch

from .base_dataloader import BaseDataLoader

logger = logging.getLogger(__name__)


class SFTDataLoader(BaseDataLoader):
    # Implicit assumption: T = max_length - 1

    def reset(self) -> None:
        self.token_shards = sorted([s for s in self.shards if "tokens" in s])
        self.label_shards = sorted([s for s in self.shards if "labels" in s])
        assert len(self.token_shards) == len(self.label_shards)
        self.current_shard = 0
        self.tokens, self.labels = self._load_shard(self.current_shard)
        self.current_example = self.B * self.rank

    def advance(self, tokens: int) -> None:
        examples = tokens // self.T
        while examples > 0:
            remaining_in_shard = self.tokens.shape[0] - self.current_example
            if examples < remaining_in_shard:
                self.current_example += examples
                examples = 0
            else:
                examples -= remaining_in_shard
                self.current_shard = (self.current_shard + 1) % len(self.token_shards)
                self.tokens, self.labels = self._load_shard(self.current_shard)
                self.current_example = self.B * self.rank

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.current_example + self.B > self.tokens.shape[0]:
            self.current_shard = (self.current_shard + 1) % len(self.token_shards)
            self.tokens, self.labels = self._load_shard(self.current_shard)
            self.current_example = self.B * self.rank
        x = self.tokens[self.current_example : self.current_example + self.B, :-1]
        y = self.labels[self.current_example : self.current_example + self.B, 1:]
        self.current_example += self.B * self.world_size
        return x, y

    def _load_shard(self, shard_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        assert shard_idx < len(self.token_shards)
        fn_tokens = self.token_shards[shard_idx]
        npt_tokens = np.load(fn_tokens)
        if self.B * self.world_size >= len(npt_tokens):
            logger.warning(
                f"Shard {shard_idx} is too small. If this warning persists across shards, "
                "it means you need to make shards bigger."
            )

        assert shard_idx < len(self.label_shards)
        fn_labels = self.label_shards[shard_idx]
        npt_labels = np.load(fn_labels)

        return torch.tensor(npt_tokens, dtype=torch.long), torch.tensor(
            npt_labels, dtype=torch.long
        )
