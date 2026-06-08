from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..constants import LABEL_IGNORE_INDEX
from ..tokenizer import TiktokenTokenizer, Tokenizer
from .base_model import BaseModel
from .config import GPT2Config


class ScaledLinear(nn.Linear):
    SAFFRON_INIT: bool = True


class CausalSelfAttention(nn.Module):
    causal: torch.Tensor

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_embd = config.n_embd
        self.n_head = config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = ScaledLinear(config.n_embd, config.n_embd)
        self.register_buffer(
            "causal",
            torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)),
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, n_embd = x.shape
        assert n_embd == self.n_embd
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.reshape(B, T, self.n_head, n_embd // self.n_head).transpose(1, 2)
        q = q.reshape(B, T, self.n_head, n_embd // self.n_head).transpose(1, 2)
        v = v.reshape(B, T, self.n_head, n_embd // self.n_head).transpose(1, 2)

        if attention_mask is not None:
            # combine (B, T) padding mask with causal mask; can't use is_causal alongside attn_mask
            assert attention_mask.shape == (B, T)
            pad = attention_mask[:, None, None, :].bool()  # (B, 1, 1, T)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=pad & self.causal[:T, :T])
        else:
            # use is_causal for performance
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().reshape(B, T, n_embd)

        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = ScaledLinear(4 * config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(BaseModel):
    config: GPT2Config  # pyright: ignore[reportIncompatibleVariableOverride]
    config_class = GPT2Config

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.config = config  # pyright: ignore[reportIncompatibleVariableOverride]

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            std = 0.02
            if hasattr(module, "SAFFRON_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0, std=std)
        if isinstance(module, nn.Linear) and module.bias is not None:  # type: ignore[redundant-expr]
            torch.nn.init.zeros_(module.bias)

    def _get_tokenizer(self) -> Tokenizer:
        return TiktokenTokenizer("gpt2")

    def forward(
        self,
        idx: torch.Tensor,
        target: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.size()
        assert self.config.block_size >= T, "block size exhausted"
        if attention_mask is not None:
            assert attention_mask.shape == (B, T)

        token_embd = self.wte(idx)  # (B, T, n_embd)
        if attention_mask is not None:
            # derive positions from the mask so (left-)padded tokens don't shift
            # real tokens: position counts from 0 over unmasked tokens; pad clamped to 0
            pos = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)  # (B, T)
            pos_embd = self.wpe(pos)  # (B, T, n_embd)
        else:
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            pos_embd = self.wpe(pos)  # (T, n_embd)
        x = token_embd + pos_embd

        for block in self.h:
            x = block(x, attention_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if target is not None:
            loss = F.cross_entropy(
                logits.reshape(B * T, self.config.vocab_size),
                target.reshape(B * T),
                ignore_index=LABEL_IGNORE_INDEX,
            )

        return logits, loss

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
        if top_p != 1.0:
            raise ValueError("Native model.generate does not support top_p yet.")
        original_device = idx.device
        # MPS has numerical issues with autoregressive generation — run on CPU
        if original_device.type == "mps":
            self.cpu()
            idx = idx.cpu()
            if attention_mask is not None:
                attention_mask = attention_mask.cpu()

        self.eval()
        try:
            pad_token_id = self.tokenizer.pad_token_id
            stop_token_ids = self.tokenizer.stop_token_ids
            # pre-fill generated positions with pad, so an early break (all rows finished)
            # leaves the unused tail as pad rather than token id 0.
            output = torch.cat(
                (
                    idx,
                    torch.full(
                        (idx.shape[0], max_new_tokens),
                        pad_token_id,
                        dtype=torch.long,
                        device=idx.device,
                    ),
                ),
                dim=1,
            )
            # extend the prompt mask with 1s for generated positions (always real tokens)
            full_mask = None
            if attention_mask is not None:
                assert attention_mask.shape == idx.shape
                full_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (idx.shape[0], max_new_tokens),
                            dtype=attention_mask.dtype,
                            device=idx.device,
                        ),
                    ),
                    dim=1,
                )
            finished = torch.zeros(idx.shape[0], dtype=torch.bool, device=idx.device)
            for col in range(idx.shape[1], output.shape[1]):
                start = max(0, col - self.config.block_size)
                window_mask = full_mask[:, start:col] if full_mask is not None else None
                logits, _ = self(output[:, start:col], attention_mask=window_mask)
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)

                topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1)
                ix = torch.multinomial(topk_probs, num_samples=1)
                next_token = torch.gather(topk_indices, -1, ix).squeeze(-1)
                next_token = next_token.masked_fill(finished, pad_token_id)
                output[:, col] = next_token
                is_stop = torch.zeros_like(finished)
                for tid in stop_token_ids:
                    is_stop = is_stop | (next_token == tid)
                finished = finished | is_stop
                if finished.all():
                    break
        finally:
            self.train()
            if original_device.type == "mps":
                self.to(original_device)
        return output.to(original_device)
