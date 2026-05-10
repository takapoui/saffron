import json
import logging
import multiprocessing as mp
import os
from collections.abc import Iterable
from typing import cast

import numpy as np
from datasets import load_dataset  # type: ignore[reportUnknownVariableType]

from ..tokenizer import Tokenizer
from .config import PrepConfig

logger = logging.getLogger(__name__)

_worker_enc: Tokenizer | None = None


def _init_worker(tokenizer_name: str) -> None:
    global _worker_enc
    _worker_enc = Tokenizer.from_name(tokenizer_name, local_files_only=True)


def _tokenize(doc: dict[str, str]) -> np.ndarray:
    enc = _worker_enc
    assert enc is not None
    dtype = np.uint16 if enc.vocab_size <= 2**16 else np.uint32
    tokens = np.array([enc.eot_token] + enc.encode(doc["text"]))
    assert (tokens >= 0).all() and (tokens < np.iinfo(dtype).max).all()
    return tokens.astype(dtype)


def load_and_tokenize_dataset(prep_config: PrepConfig) -> None:
    enc = Tokenizer.from_name(prep_config.tokenizer)
    fw = cast(
        Iterable[dict[str, str]],
        load_dataset(
            prep_config.dataset,
            prep_config.name if prep_config.name else None,
            split=prep_config.dataset_split,
        ),
    )
    os.makedirs(prep_config.data_root, exist_ok=True)
    dtype = "uint16" if enc.vocab_size <= 2**16 else "uint32"
    meta_path = os.path.join(prep_config.data_root, "meta.json")
    with open(meta_path, "w") as f:
        json.dump({"tokenizer": prep_config.tokenizer}, f)

    def _write_to_file(arr: np.ndarray, shard_index: int) -> None:
        # Use first documents as validation
        sp = "val" if shard_index < prep_config.num_validation_shards else "train"
        filename = os.path.join(prep_config.data_root, f"{sp}_{shard_index:06d}")
        np.save(filename, arr)

    nprocs = max(1, (os.cpu_count() or 2) // 2)
    with mp.Pool(nprocs, initializer=_init_worker, initargs=(prep_config.tokenizer,)) as pool:
        shard_index = 0
        all_tokens_np = np.empty((prep_config.shard_size,), dtype=dtype)
        token_count = 0

        for tokens in pool.imap(_tokenize, fw, chunksize=16):
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
