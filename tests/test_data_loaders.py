"""Shape and correctness tests for PretrainDataLoader and SFTDataLoader."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest

from saffron.constants import LABEL_IGNORE_INDEX
from saffron.data import DataConfig, LoaderType, PretrainDataLoader, SFTDataLoader
from saffron.helpers import RunConfig

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def run_config() -> RunConfig:
    return RunConfig(
        device="cpu",
        device_type="cpu",
        use_ddp=False,
        ddp_rank=0,
        ddp_local_rank=0,
        ddp_world_size=1,
    )


# ---------------------------------------------------------------------------
# PretrainDataLoader
# ---------------------------------------------------------------------------

PretrainDirFixture = tuple[Path, int, int]  # (root, B, T)


@pytest.fixture
def pretrain_data_dir() -> Generator[PretrainDirFixture, None, None]:
    """One shard of 50 tokens — enough to exercise shard-wrap logic."""
    B, T = 2, 4

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        split_dir = root / "train"
        split_dir.mkdir()
        (root / "meta.json").write_text(json.dumps({"tokenizer": "gpt2"}))

        # 50 tokens wraps several times for B=2, T=4 (need 9 per call)
        tokens = np.arange(50, dtype=np.int32)
        np.save(split_dir / "000000.npy", tokens)

        yield root, B, T


def test_pretrain_dataloader_wraps_without_bad_shape(
    pretrain_data_dir: PretrainDirFixture, run_config: RunConfig
) -> None:
    """next_batch() must always return (B, T) tensors, even across shard wraps."""
    root, B, T = pretrain_data_dir
    cfg = DataConfig(
        data_root=root, batch_size=B, seq_len=T, tokenizer="gpt2", loader_type=LoaderType.PRETRAIN
    )
    loader = PretrainDataLoader(data_config=cfg, run_config=run_config, split="train")

    for _ in range(12):  # >1 full pass over the 50-token shard
        x, y = loader.next_batch()
        assert x.shape == (B, T), f"x.shape={x.shape}"
        assert y.shape == (B, T), f"y.shape={y.shape}"


# ---------------------------------------------------------------------------
# SFTDataLoader — basic batch shapes (from original test_sft_prep.py)
# ---------------------------------------------------------------------------

SFTDataDirFixture = tuple[Path, int, int]  # (root, MAX_LEN, N_EXAMPLES)


@pytest.fixture
def sft_data_dir() -> Generator[SFTDataDirFixture, None, None]:
    """Minimal SFT dataset: 8 examples, MAX_LEN=16, prompt at first 6 tokens."""
    MAX_LEN = 16
    N_EXAMPLES = 8
    VOCAB_SIZE = 1000

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        split_dir = root / "train"
        split_dir.mkdir()

        (root / "meta.json").write_text(json.dumps({"tokenizer": "gpt2"}))

        # prompt is first 6 tokens, answer is next 10
        tokens = np.random.randint(1, VOCAB_SIZE, size=(N_EXAMPLES, MAX_LEN), dtype=np.int32)
        labels = np.full((N_EXAMPLES, MAX_LEN), LABEL_IGNORE_INDEX, dtype=np.int32)
        labels[:, 6:] = tokens[:, 6:]

        np.save(split_dir / "000000_tokens.npy", tokens)
        np.save(split_dir / "000000_labels.npy", labels)

        yield root, MAX_LEN, N_EXAMPLES


def test_sft_dataloader_batch_shapes(
    sft_data_dir: SFTDataDirFixture, run_config: RunConfig
) -> None:
    root, max_len, _ = sft_data_dir
    data_config = DataConfig(
        data_root=root,
        batch_size=2,
        seq_len=max_len - 1,
        tokenizer="gpt2",
        loader_type=LoaderType.SFT,
    )
    loader = SFTDataLoader(data_config=data_config, run_config=run_config, split="train")
    x, y = loader.next_batch()
    assert x.shape == (2, max_len - 1)
    assert y.shape == (2, max_len - 1)


def test_sft_dataloader_labels_masked_for_prompt(
    sft_data_dir: SFTDataDirFixture, run_config: RunConfig
) -> None:
    root, max_len, _ = sft_data_dir
    data_config = DataConfig(
        data_root=root,
        batch_size=2,
        seq_len=max_len - 1,
        tokenizer="gpt2",
        loader_type=LoaderType.SFT,
    )
    loader = SFTDataLoader(data_config=data_config, run_config=run_config, split="train")
    _, y = loader.next_batch()
    # first 5 columns of y correspond to labels[:, 1:6] which are LABEL_IGNORE_INDEX
    assert (y[:, :5] == LABEL_IGNORE_INDEX).all()
    assert (y[:, 5:] != LABEL_IGNORE_INDEX).any()


def test_sft_dataloader_wraps_shards(
    sft_data_dir: SFTDataDirFixture, run_config: RunConfig
) -> None:
    root, max_len, n_examples = sft_data_dir
    data_config = DataConfig(
        data_root=root,
        batch_size=2,
        seq_len=max_len - 1,
        tokenizer="gpt2",
        loader_type=LoaderType.SFT,
    )
    loader = SFTDataLoader(data_config=data_config, run_config=run_config, split="train")
    for _ in range(n_examples + 2):
        loader.next_batch()


# ---------------------------------------------------------------------------
# SFTDataLoader — shard-boundary shape stability
# ---------------------------------------------------------------------------

SFTWrapDirFixture = tuple[Path, int, int, int]  # (root, MAX_LEN, N_EXAMPLES, B)


@pytest.fixture
def sft_wrap_dir() -> Generator[SFTWrapDirFixture, None, None]:
    """8 examples in one shard.  Batch-size 3 forces a mid-shard wrap."""
    MAX_LEN = 10
    N_EXAMPLES = 8
    B = 3
    VOCAB_SIZE = 500
    PROMPT_TOKS = 4  # first 4 positions are prompt → masked in labels

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        split_dir = root / "train"
        split_dir.mkdir()
        (root / "meta.json").write_text(json.dumps({"tokenizer": "gpt2"}))

        tokens = np.random.randint(1, VOCAB_SIZE, size=(N_EXAMPLES, MAX_LEN), dtype=np.int32)
        labels = np.full((N_EXAMPLES, MAX_LEN), LABEL_IGNORE_INDEX, dtype=np.int32)
        labels[:, PROMPT_TOKS:] = tokens[:, PROMPT_TOKS:]

        np.save(split_dir / "000000_tokens.npy", tokens)
        np.save(split_dir / "000000_labels.npy", labels)

        yield root, MAX_LEN, N_EXAMPLES, B


def test_sft_dataloader_wraps_without_bad_shape(
    sft_wrap_dir: SFTWrapDirFixture, run_config: RunConfig
) -> None:
    """next_batch() must return consistent (B, T) shapes across shard boundaries
    and after the loader wraps back to the start."""
    root, max_len, n_examples, B = sft_wrap_dir
    T = max_len - 1  # SFTDataLoader slices off one token

    cfg = DataConfig(
        data_root=root,
        batch_size=B,
        seq_len=T,
        tokenizer="gpt2",
        loader_type=LoaderType.SFT,
    )
    loader = SFTDataLoader(data_config=cfg, run_config=run_config, split="train")

    # Call enough times to wrap the 8-example shard at least once with B=3
    for _ in range(n_examples + 2):
        x, y = loader.next_batch()
        assert x.shape == (B, T), f"x.shape={x.shape}"
        assert y.shape == (B, T), f"y.shape={y.shape}"
