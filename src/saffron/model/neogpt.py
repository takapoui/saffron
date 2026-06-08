from __future__ import annotations

from typing import TypeAlias

import torch
import torch.nn.functional as F
from torch import nn

from ..constants import LABEL_IGNORE_INDEX
from ..tokenizer import TiktokenTokenizer, Tokenizer
from .base_model import BaseModel
from .config import NeoGPTConfig

KVCache: TypeAlias = tuple[torch.Tensor, torch.Tensor]


class ScaledLinear(nn.Linear):
    SAFFRON_INIT: bool = True


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    # (x0, x1, x2, x3) -> (-x2, -x3, x0, x1)
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


class RotaryEmbedding(nn.Module):
    # caches sins and cosines once and resue them
    cos_cache: torch.Tensor
    sin_cache: torch.Tensor

    def __init__(self, dim: int, base: int, max_seq_len: int) -> None:
        super().__init__()
        # theta_i = 1 / (base ^ (2i / dim)),  shape: (dim/2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer("cos_cache", emb.cos())
        self.register_buffer("sin_cache", emb.sin())

    def apply_rope(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        T = x.shape[-2]
        cos, sin = (
            self.cos_cache[start_pos : start_pos + T],
            self.sin_cache[start_pos : start_pos + T],
        )
        cos, sin = cos[None, None, :, :], sin[None, None, :, :]  # (T, d_head) -> (1, 1, T, d_head)
        return x * cos + _rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    causal: torch.Tensor

    def __init__(self, config: NeoGPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = self.n_embd // self.n_head

        self.c_attn = nn.Linear(
            config.n_embd, config.n_embd + 2 * self.n_kv_head * self.head_dim, bias=False
        )
        self.c_proj = ScaledLinear(config.n_embd, config.n_embd, bias=False)
        self.rotary = RotaryEmbedding(
            dim=config.n_embd // config.n_head,
            base=config.rope_base,
            max_seq_len=config.block_size,
        )
        self.register_buffer(
            "causal",
            torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_kv: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        B, T, n_embd = x.shape
        assert n_embd == self.n_embd
        q, k, v = self.c_attn(x).split(
            [n_embd, self.n_kv_head * self.head_dim, self.n_kv_head * self.head_dim], dim=2
        )

        q = q.reshape(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        past_len = past_kv[0].shape[2] if past_kv is not None else 0
        q = self.rotary.apply_rope(q, start_pos=past_len)
        k = self.rotary.apply_rope(k, start_pos=past_len)
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        present_kv = (k, v)

        is_prefill = past_kv is None
        if attention_mask is not None:
            # attention_mask is a validity mask over all key positions (B, Lk),
            # where Lk = past_len + T. Combined with causal; can't use is_causal alongside attn_mask
            assert attention_mask.shape == (B, k.shape[2])
            pad = attention_mask[:, None, None, :].bool()  # (B, 1, 1, Lk)
            causal = self.causal[past_len : past_len + T, : k.shape[2]]
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=pad & causal, enable_gqa=True)
        else:
            # prefill: square causal mask. decode: single query sees only cached past (no mask)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=is_prefill, enable_gqa=True)
        y = y.transpose(1, 2).contiguous().reshape(B, T, n_embd)

        y = self.c_proj(y)
        return y, present_kv


class MLP(nn.Module):
    def __init__(self, config: NeoGPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.mlp_hidden_dim, bias=False)
        self.gate = nn.Linear(config.n_embd, config.mlp_hidden_dim, bias=False)
        self.c_proj = ScaledLinear(config.mlp_hidden_dim, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = self.c_fc(x)
        gate = self.gate(x)
        x = F.silu(gate) * up
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: NeoGPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_kv: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        att, present_kv = self.attn(self.ln_1(x), attention_mask=attention_mask, past_kv=past_kv)
        x = x + att
        x = x + self.mlp(self.ln_2(x))
        return x, present_kv


class NeoGPT(BaseModel):
    config: NeoGPTConfig  # pyright: ignore[reportIncompatibleVariableOverride]
    config_class = NeoGPTConfig

    def __init__(self, config: NeoGPTConfig) -> None:
        super().__init__()
        self.config = config  # pyright: ignore[reportIncompatibleVariableOverride]

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            std = 0.02
            if hasattr(module, "SAFFRON_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0, std=std)

    def _get_tokenizer(self) -> Tokenizer:
        return TiktokenTokenizer("gpt2")

    def forward(
        self,
        idx: torch.Tensor,
        target: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits, loss, _ = self.forward_with_cache(
            idx=idx,
            target=target,
            attention_mask=attention_mask,
        )
        return logits, loss

    def forward_with_cache(
        self,
        idx: torch.Tensor,
        target: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_kvs: list[KVCache | None] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[KVCache | None]]:
        B, T = idx.size()
        assert self.config.block_size >= T, "block size exhausted"

        x = self.wte(idx)  # (B, T, n_embd)

        kvs: list[KVCache | None] = [None] * self.config.n_layer if past_kvs is None else past_kvs

        new_kvs: list[KVCache | None] = []
        for i, block in enumerate(self.h):
            x, present_kv = block(x, attention_mask=attention_mask, past_kv=kvs[i])
            if use_cache:
                new_kvs.append(present_kv)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if target is not None:
            loss = F.cross_entropy(
                logits.reshape(B * T, self.config.vocab_size),
                target.reshape(B * T),
                ignore_index=LABEL_IGNORE_INDEX,
            )

        return logits, loss, new_kvs

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
        assert idx.shape[1] + max_new_tokens <= self.config.block_size, "Exceeds block_size"
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

            finished = torch.zeros(idx.shape[0], dtype=torch.bool, device=idx.device)

            full_mask: torch.Tensor | None = None
            if attention_mask is not None:
                assert attention_mask.shape == idx.shape
                full_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            idx.shape[0],
                            max_new_tokens,
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ],
                    dim=1,
                )

            prompt_mask = full_mask[:, : idx.shape[1]] if full_mask is not None else None
            kvs: list[KVCache | None] | None = None
            logits, _, kvs = self.forward_with_cache(
                idx, attention_mask=prompt_mask, past_kvs=kvs, use_cache=True
            )

            for col in range(idx.shape[1], output.shape[1]):
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

                step_mask = full_mask[:, : col + 1] if full_mask is not None else None
                logits, _, kvs = self.forward_with_cache(
                    next_token[:, None], attention_mask=step_mask, past_kvs=kvs, use_cache=True
                )
        finally:
            self.train()
            if original_device.type == "mps":
                self.to(original_device)
        return output.to(original_device)
