# type: ignore  # transformers has no type stubs
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from ..tokenizer import HFTokenizer, Tokenizer
from .base_model import BaseModel
from .config import BaseConfig


@dataclass
class HFConfig(BaseConfig):
    hf_model_name: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HFConfig:
        hf_config = AutoConfig.from_pretrained(d["hf_model_name"])
        return cls(
            vocab_size=hf_config.vocab_size,
            n_embd=hf_config.hidden_size,
            block_size=hf_config.max_position_embeddings,
            hf_model_name=d["hf_model_name"],
        )


class HFModel(BaseModel):
    config: HFConfig
    config_class = HFConfig

    def supports_compile(self, device_type: str) -> bool:
        return device_type == "cuda"

    def __init__(self, config: HFConfig) -> None:
        super().__init__()
        self.config = config
        self.hf_model = AutoModelForCausalLM.from_pretrained(config.hf_model_name)
        self._tokenizer = HFTokenizer(config.hf_model_name)

    def get_tokenizer(self) -> Tokenizer:
        return self._tokenizer

    def forward(
        self,
        idx: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.hf_model(input_ids=idx).logits
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return logits, loss
