# type: ignore  # transformers has no type stubs
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from ..constants import LABEL_IGNORE_INDEX
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

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1,
        top_k: int = 50,
        stop_token_ids: list[int] | None = None,
    ) -> torch.Tensor:
        # Use HF's built-in generate which supports KV cache
        eos_token_id = stop_token_ids if stop_token_ids is not None else None
        out = self.hf_model.generate(
            input_ids=idx,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0 and top_k > 1,
            temperature=temperature if temperature > 0 and top_k > 1 else None,
            top_k=top_k if top_k > 1 else None,
            eos_token_id=eos_token_id,
            pad_token_id=eos_token_id[0] if eos_token_id else None,
        )
        return out

    def forward(
        self,
        idx: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.hf_model(input_ids=idx).logits
        loss = None
        if target is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target.view(-1),
                ignore_index=LABEL_IGNORE_INDEX,
            )
        return logits, loss
