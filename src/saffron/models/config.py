from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
