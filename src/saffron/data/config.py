from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    data_root: Path
    batch_size: int
    seq_len: int
    tokenizer: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DataConfig:
        return cls(
            data_root=Path(d["data_root"]),
            batch_size=d["batch_size"],
            seq_len=d["seq_len"],
            tokenizer=d["tokenizer"],
        )


@dataclass
class PrepConfig:
    data_root: Path
    dataset: str
    name: str
    shard_size: int
    tokenizer: str
    dataset_split: str
    num_validation_shards: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PrepConfig:
        return cls(
            data_root=Path(d["data_root"]),
            dataset=d["dataset"],
            name=d["name"],
            shard_size=d["shard_size"],
            tokenizer=d["tokenizer"],
            dataset_split=d["dataset_split"],
            num_validation_shards=d["num_validation_shards"],
        )
