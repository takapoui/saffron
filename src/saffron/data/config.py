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
class BasePrepConfig:
    data_root: Path
    dataset: str
    name: str
    tokenizer: str
    dataset_split: str

    @classmethod
    def _base_kwargs(cls, d: dict[str, Any]) -> dict[str, Any]:
        return {
            "data_root": Path(d["data_root"]),
            "dataset": d["dataset"],
            "name": d["name"],
            "tokenizer": d["tokenizer"],
            "dataset_split": d["dataset_split"],
        }


@dataclass
class PretrainPrepConfig(BasePrepConfig):
    shard_size: int

    num_validation_shards: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PretrainPrepConfig:
        return cls(
            **cls._base_kwargs(d),
            shard_size=d["shard_size"],
            num_validation_shards=d["num_validation_shards"],
        )


@dataclass
class SFTPrepConfig(BasePrepConfig):
    examples_per_shard: int
    max_length: int
    system_prompt: str | None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SFTPrepConfig:
        return cls(
            **cls._base_kwargs(d),
            examples_per_shard=d["examples_per_shard"],
            max_length=d["max_length"],
            system_prompt=d.get("system_prompt"),
        )
