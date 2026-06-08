from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from transformers import AutoConfig  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from typing import Self


@dataclass
class BaseConfig(ABC):
    vocab_size: int
    n_embd: int
    block_size: int

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        pass


@dataclass
class GPT2Config(BaseConfig):
    n_layer: int
    n_head: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GPT2Config:
        return cls(
            vocab_size=d["vocab_size"],
            n_embd=d["n_embd"],
            block_size=d["block_size"],
            n_layer=d["n_layer"],
            n_head=d["n_head"],
        )


@dataclass
class NeoGPTConfig(BaseConfig):
    n_layer: int
    n_head: int
    rope_base: int
    mlp_hidden_dim: int
    n_kv_head: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NeoGPTConfig:
        return cls(
            vocab_size=d["vocab_size"],
            n_embd=d["n_embd"],
            block_size=d["block_size"],
            n_layer=d["n_layer"],
            n_head=d["n_head"],
            rope_base=d["rope_base"],
            mlp_hidden_dim=d["mlp_hidden_dim"],
            n_kv_head=d["n_kv_head"],
        )


@dataclass
class HFConfig(BaseConfig):
    hf_model_name: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HFConfig:
        hf_config = AutoConfig.from_pretrained(d["hf_model_name"])  # type: ignore[reportUnknownMemberType]
        return cls(
            vocab_size=int(hf_config.vocab_size),  # type: ignore[reportUnknownMemberType]
            n_embd=int(hf_config.hidden_size),  # type: ignore[reportUnknownMemberType]
            block_size=int(hf_config.max_position_embeddings),  # type: ignore[reportUnknownMemberType]
            hf_model_name=d["hf_model_name"],
        )
