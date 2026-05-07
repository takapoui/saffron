import logging
import multiprocessing as mp
import os
from collections.abc import Iterable
from functools import partial
from typing import cast

import numpy as np
import tiktoken
import torch
from datasets import load_dataset  # type: ignore[reportUnknownVariableType]

from .config import DataConfig, PrepConfig, RunConfig

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


def _tokenize(doc: dict[str, str], enc: tiktoken.Encoding) -> np.ndarray:
    eot = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
    tokens = np.array([eot] + enc.encode_ordinary(doc["text"]))
    assert (tokens >= 0).all() and (tokens < 2**16).all()
    return tokens.astype(np.uint16)


def load_and_tokenize_dataset(prep_config: PrepConfig) -> None:
    enc = tiktoken.get_encoding(prep_config.tokenizer)
    fw = cast(
        Iterable[dict[str, str]],
        load_dataset(
            prep_config.dataset,
            prep_config.name if prep_config.name else None,
            split=prep_config.dataset_split,
        ),
    )
    os.makedirs(prep_config.data_root, exist_ok=True)

    def _write_to_file(arr: np.ndarray, shard_index: int) -> None:
        # Use first documents as validation
        sp = "val" if shard_index < prep_config.num_validation_shards else "train"
        filename = os.path.join(prep_config.data_root, f"{sp}_{shard_index:06d}")
        np.save(filename, arr)

    nprocs = max(1, (os.cpu_count() or 2) // 2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((prep_config.shard_size,), dtype=np.uint16)
        token_count = 0

        for tokens in pool.imap(partial(_tokenize, enc=enc), fw, chunksize=16):
            if token_count + len(tokens) < prep_config.shard_size:
                all_tokens_np[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)
            else:
                remainder = prep_config.shard_size - token_count
                all_tokens_np[token_count:] = tokens[:remainder]
                _write_to_file(all_tokens_np, shard_index)

                shard_index += 1
                all_tokens_np[: len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder
        if token_count > 0:
            _write_to_file(all_tokens_np[:token_count], shard_index)
