from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Self

import torch
from torch import nn

from ..tokenizer import Tokenizer
from .config import BaseConfig


class BaseModel(nn.Module, ABC):
    config: BaseConfig
    config_class: type[BaseConfig]
    _tokenizer: Tokenizer | None = None

    def supports_compile(self, device_type: str) -> bool:
        return True

    @abstractmethod
    def _get_tokenizer(self) -> Tokenizer:
        pass

    @property
    def tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            self._tokenizer = self._get_tokenizer()
        return self._tokenizer

    @abstractmethod
    def forward(
        self,
        idx: torch.Tensor,
        target: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError

    def save_to_file(self, fn: Path) -> None:
        torch.save({"model_config": self.config, "model_dict": self.state_dict()}, fn)

    @classmethod
    def load_from_file(cls, fn: Path, device: str = "cpu") -> Self:
        checkpoint = torch.load(fn, weights_only=False, map_location=device)
        model = cls(checkpoint["model_config"])
        state_dict = checkpoint["model_dict"]
        # torch.compile wraps keys with "_orig_mod." prefix — strip it
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        return model.to(device)

    @abstractmethod
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,  # already tokenized and on device
        max_new_tokens: int,
        temperature: float = 1,
        top_k: int = 50,
        top_p: float = 1.0,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pass
