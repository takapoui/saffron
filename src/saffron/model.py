from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig


class ScaledLinear(nn.Linear):
    SAFFRON_INIT: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_embd = config.n_embd
        self.n_head = config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = ScaledLinear(config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, n_embd = x.shape
        assert n_embd == self.n_embd
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.reshape(B, T, self.n_head, n_embd // self.n_head).transpose(1, 2)
        q = q.reshape(B, T, self.n_head, n_embd // self.n_head).transpose(1, 2)
        v = v.reshape(B, T, self.n_head, n_embd // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().reshape(B, T, n_embd)

        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
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
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Model(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

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

    def forward(
        self,
        idx: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.size()
        assert self.config.block_size >= T, "block size exhausted"

        token_embd = self.wte(idx)  # (B, T, n_embd)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_embd = self.wpe(pos)  # (T, n_embd)
        x = token_embd + pos_embd

        for block in self.h:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if target is not None:
            loss = F.cross_entropy(
                logits.reshape(B * T, self.config.vocab_size),
                target.reshape(B * T),
            )

        return logits, loss

    def save_to_file(self, fn: Path) -> None:
        torch.save({"config": self.config, "model": self.state_dict()}, fn)

    @classmethod
    def load_from_file(cls, fn: Path) -> Model:
        checkpoint = torch.load(fn, weights_only=False)
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["model"])
        return model

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,  # already tokenized and on device
        max_new_tokens: int,
        temperature: float = 1,
        top_k: int = 50,
    ) -> torch.Tensor:
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
        return output
