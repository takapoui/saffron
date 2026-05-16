# type: ignore  # transformers has no type stubs
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from ..constants import LABEL_IGNORE_INDEX
from ..tokenizer import HFTokenizer
from .base_model import BaseModel
from .config import HFConfig


class HFModel(BaseModel):
    config: HFConfig
    config_class = HFConfig

    def supports_compile(self, device_type: str) -> bool:
        return device_type == "cuda"

    def __init__(self, config: HFConfig) -> None:
        super().__init__()
        self.config = config
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            config.hf_model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

    def _get_tokenizer(self) -> HFTokenizer:
        return HFTokenizer(self.config.hf_model_name)

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1,
        top_k: int = 50,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Use HF's built-in generate which supports KV cache
        out = self.hf_model.generate(
            input_ids=idx,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0 and top_k > 1,
            temperature=temperature if temperature > 0 and top_k > 1 else None,
            top_k=top_k if top_k > 1 else None,
            eos_token_id=self.tokenizer.stop_token_ids,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return out

    def forward(
        self,
        idx: torch.Tensor,
        target: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.hf_model(input_ids=idx, attention_mask=attention_mask).logits
        loss = None
        if target is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target.view(-1),
                ignore_index=LABEL_IGNORE_INDEX,
            )
        return logits, loss
