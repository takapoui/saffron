from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Self

import torch
import torch.nn.functional as F
from torch import nn

from ..tokenizer import Tokenizer
from .config import BaseConfig


class BaseModel(nn.Module, ABC):
    config: BaseConfig
    config_class: type[BaseConfig]

    def supports_compile(self, device_type: str) -> bool:
        return True

    @abstractmethod
    def get_tokenizer(self) -> Tokenizer:
        pass

    @abstractmethod
    def forward(
        self,
        idx: torch.Tensor,
        target: torch.Tensor | None = None,
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

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,  # already tokenized and on device
        max_new_tokens: int,
        temperature: float = 1,
        top_k: int = 50,
    ) -> torch.Tensor:
        original_device = idx.device
        # MPS has numerical issues with autoregressive generation — run on CPU
        if original_device.type == "mps":
            self.cpu()
            idx = idx.cpu()

        self.eval()
        try:
            output = torch.cat(
                (
                    idx,
                    torch.zeros(
                        (idx.shape[0], max_new_tokens), dtype=torch.long, device=idx.device
                    ),
                ),
                dim=1,
            )
            for col in range(idx.shape[1], output.shape[1]):
                logits, _ = self(output[:, max(0, col - self.config.block_size) : col])
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)

                topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1)
                ix = torch.multinomial(topk_probs, num_samples=1)
                output[:, col] = torch.gather(topk_indices, -1, ix).squeeze(-1)
        finally:
            self.train()
            if original_device.type == "mps":
                self.to(original_device)
        return output.to(original_device)
