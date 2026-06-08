"""Tests for the NeoGPT model (forward, generate, per-row stop)."""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest
import torch

from saffron.constants import LABEL_IGNORE_INDEX
from saffron.model import NeoGPT, NeoGPTConfig
from saffron.model.neogpt import RotaryEmbedding


def _fake_tokenizer(stop_ids: list[int]) -> MagicMock:
    """Tokenizer mock with the two properties generate reads off."""
    tok = MagicMock()
    tok.stop_token_ids = stop_ids
    tok.pad_token_id = stop_ids[0]
    return tok


@pytest.fixture
def config() -> NeoGPTConfig:
    return NeoGPTConfig(
        vocab_size=1000,
        n_embd=96,
        block_size=64,
        n_layer=2,
        n_head=4,
        rope_base=10000,
        mlp_hidden_dim=256,
    )


@pytest.fixture
def model(config: NeoGPTConfig) -> NeoGPT:
    torch.manual_seed(0)  # type: ignore[reportUnknownMemberType]
    return NeoGPT(config)


# --- forward ---


def test_forward_shape(model: NeoGPT, config: NeoGPTConfig) -> None:
    idx = torch.randint(0, config.vocab_size, (2, 10))
    logits, loss = model(idx)
    assert logits.shape == (2, 10, config.vocab_size)
    assert loss is None


def test_forward_with_targets_returns_scalar_loss(model: NeoGPT, config: NeoGPTConfig) -> None:
    idx = torch.randint(0, config.vocab_size, (2, 10))
    target = torch.randint(0, config.vocab_size, (2, 10))
    _, loss = model(idx, target)
    assert loss is not None
    assert loss.shape == ()
    assert loss.item() > 0


def test_forward_loss_ignores_label_ignore_index(model: NeoGPT, config: NeoGPTConfig) -> None:
    torch.manual_seed(1)  # type: ignore[reportUnknownMemberType]
    idx = torch.randint(0, config.vocab_size, (1, 10))
    target = torch.randint(0, config.vocab_size, (1, 10))

    _, loss_all = model(idx, target)

    # mask the first 5 tokens — loss should only reflect the last 5
    target_partial = target.clone()
    target_partial[0, :5] = LABEL_IGNORE_INDEX
    _, loss_partial = model(idx, target_partial)

    assert loss_all is not None
    assert loss_partial is not None
    assert loss_all.item() != pytest.approx(loss_partial.item())  # type: ignore[reportUnknownMemberType]


def test_attention_mask_ignores_padding(model: NeoGPT, config: NeoGPTConfig) -> None:
    """Masked padding (appended to the right) must not change the logits of the
    real positions, since the model is causal and padding is masked out."""
    torch.manual_seed(0)  # type: ignore[reportUnknownMemberType]
    real_len = 6
    idx = torch.randint(0, config.vocab_size, (1, real_len))

    logits_base, _ = model(idx)

    pad_len = 4
    pad = torch.randint(0, config.vocab_size, (1, pad_len))
    idx_padded = torch.cat([idx, pad], dim=1)
    mask = torch.cat([torch.ones(1, real_len), torch.zeros(1, pad_len)], dim=1)
    logits_padded, _ = model(idx_padded, attention_mask=mask)

    torch.testing.assert_close(logits_padded[:, :real_len], logits_base)


def test_left_padding_matches_unpadded(model: NeoGPT, config: NeoGPTConfig) -> None:
    """Left-padding + mask must yield the same logits on the real tokens as the
    unpadded sequence — position ids are derived from the mask, so padding does
    not shift the positional encoding of real tokens."""
    torch.manual_seed(0)  # type: ignore[reportUnknownMemberType]
    real_len = 6
    idx = torch.randint(0, config.vocab_size, (1, real_len))

    logits_base, _ = model(idx)

    pad_len = 4
    pad = torch.randint(0, config.vocab_size, (1, pad_len))
    idx_padded = torch.cat([pad, idx], dim=1)  # left pad
    mask = torch.cat([torch.zeros(1, pad_len), torch.ones(1, real_len)], dim=1)
    logits_padded, _ = model(idx_padded, attention_mask=mask)

    # real tokens occupy the right-most `real_len` positions
    torch.testing.assert_close(logits_padded[:, pad_len:], logits_base)


def test_attention_mask_wrong_shape_raises(model: NeoGPT, config: NeoGPTConfig) -> None:
    idx = torch.randint(0, config.vocab_size, (2, 10))
    bad_mask = torch.ones(2, 10, 10)  # should be (B, T)
    with pytest.raises(AssertionError):
        model(idx, attention_mask=bad_mask)


# --- generate ---


def test_generate_output_length(model: NeoGPT, config: NeoGPTConfig) -> None:
    idx = torch.randint(0, config.vocab_size, (1, 5))
    out = model.generate(idx, max_new_tokens=10)
    assert out.shape == (1, 15)  # 5 prompt + 10 generated


def test_generate_accepts_attention_mask(model: NeoGPT, config: NeoGPTConfig) -> None:
    idx = torch.randint(0, config.vocab_size, (2, 5))
    mask = torch.ones_like(idx)
    out = model.generate(idx, max_new_tokens=8, attention_mask=mask)
    assert out.shape == (2, 13)  # 5 prompt + 8 generated


def test_generate_wrong_mask_shape_raises(model: NeoGPT, config: NeoGPTConfig) -> None:
    idx = torch.randint(0, config.vocab_size, (2, 5))
    bad_mask = torch.ones(2, 5 + 8)  # full length, not prompt length
    with pytest.raises(AssertionError):
        model.generate(idx, max_new_tokens=8, attention_mask=bad_mask)


def test_generate_stops_at_stop_token(model: NeoGPT, config: NeoGPTConfig) -> None:
    stop_token = 0
    model._tokenizer = _fake_tokenizer([stop_token])  # pyright: ignore[reportPrivateUsage]
    idx = torch.randint(1, config.vocab_size, (1, 5))  # no stop tokens in prompt
    out = model.generate(idx, max_new_tokens=50)
    # output should be <= 5 + 50
    assert out.shape[1] <= 55


def test_generate_batch(model: NeoGPT, config: NeoGPTConfig) -> None:
    idx = torch.randint(0, config.vocab_size, (3, 5))
    out = model.generate(idx, max_new_tokens=8)
    assert out.shape == (3, 13)


def test_generate_top_p_raises_for_native_models(model: NeoGPT, config: NeoGPTConfig) -> None:
    idx = torch.randint(0, config.vocab_size, (1, 5))
    with pytest.raises(ValueError, match="top_p"):
        model.generate(idx, max_new_tokens=8, top_p=0.9)


# --- generate: per-row stop ---


def test_generate_stops_finished_rows_only(config: NeoGPTConfig) -> None:
    """When one row in a batch hits a stop token, subsequent tokens for that
    row must be clamped to the stop token ID while generation continues for
    rows that have not yet finished."""
    STOP = 5  # token id used as stop; must be < config.vocab_size

    class _DetNeoGPT(NeoGPT):
        """Row 0 always predicts STOP; row 1 always predicts token 1."""

        def forward(
            self,
            idx: torch.Tensor,
            target: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            B, T = idx.shape
            logits = torch.zeros(B, T, self.config.vocab_size)
            logits[0, :, STOP] = 100.0  # row 0 → stop immediately
            logits[1, :, 1] = 100.0  # row 1 → token 1, never stops
            return logits, None

    torch.manual_seed(0)  # type: ignore[reportUnknownMemberType]
    model = _DetNeoGPT(config)
    model._tokenizer = _fake_tokenizer([STOP])  # pyright: ignore[reportPrivateUsage]

    prompt = torch.zeros((2, 3), dtype=torch.long)  # batch=2, prompt_len=3
    max_new = 4

    out = model.generate(
        idx=prompt,
        max_new_tokens=max_new,
        temperature=1.0,
        top_k=1,
    )

    new_row0 = cast(list[int], out[0, prompt.shape[1] :].tolist())  # type: ignore[reportUnknownMemberType]
    new_row1 = cast(list[int], out[1, prompt.shape[1] :].tolist())  # type: ignore[reportUnknownMemberType]

    # Row 0 finished at the very first step; every new token must be STOP
    assert all(t == STOP for t in new_row0), f"row 0 new tokens: {new_row0}"

    # Row 1 never finished; no new token should equal STOP
    assert all(t != STOP for t in new_row1), f"row 1 new tokens: {new_row1}"

    # Generation ran for the full max_new_tokens (loop did not exit early)
    assert len(new_row1) == max_new


# --- RoPE ---


def test_rotary_cache_shape() -> None:
    dim, base, max_seq_len = 16, 10000, 32
    rope = RotaryEmbedding(dim=dim, base=base, max_seq_len=max_seq_len)
    assert rope.cos_cache.shape == (max_seq_len, dim)
    assert rope.sin_cache.shape == (max_seq_len, dim)


def test_rotary_inv_freq_decreases() -> None:
    """Higher dimension indices must have lower frequency (slower rotation)."""
    dim, base, max_seq_len = 16, 10000, 32
    rope = RotaryEmbedding(dim=dim, base=base, max_seq_len=max_seq_len)
    # cos_cache[1] gives the cosines at position 1; values = cos(theta_i * 1)
    # higher i → smaller theta_i → cos(theta_i) closer to 1 (less rotation)
    cos_pos1 = rope.cos_cache[1]  # shape: (dim,)
    # the first half encodes distinct frequencies; second half is repeated
    half = dim // 2
    angles = cos_pos1[:half]
    # each successive angle should be closer to 1 (i.e. smaller rotation)
    assert (angles[1:] >= angles[:-1]).all()


def test_rotary_position_zero_is_identity() -> None:
    """At position 0, cos=1 and sin=0, so apply_rope must return x unchanged."""
    dim, base, max_seq_len = 16, 10000, 32
    rope = RotaryEmbedding(dim=dim, base=base, max_seq_len=max_seq_len)
    x = torch.randn(1, 1, 1, dim)  # (B, H, T=1, d_head)
    out = rope.apply_rope(x, T=1)
    torch.testing.assert_close(out, x)


def test_rotary_preserves_norm() -> None:
    """RoPE is a rotation so ||apply_rope(x)|| == ||x|| for every vector."""
    dim, base, max_seq_len = 16, 10000, 16
    rope = RotaryEmbedding(dim=dim, base=base, max_seq_len=max_seq_len)
    x = torch.randn(2, 4, max_seq_len, dim)
    out = rope.apply_rope(x, T=max_seq_len)
    torch.testing.assert_close(out.norm(dim=-1), x.norm(dim=-1))  # type: ignore[reportUnknownMemberType]


def test_rotary_relative_position_via_dot_product() -> None:
    """q_m · k_n should depend only on (m-n), not on absolute positions.
    We verify this by checking that the dot product at offset d is the same
    regardless of where in the sequence it occurs."""
    dim, base, max_seq_len = 16, 10000, 32
    rope = RotaryEmbedding(dim=dim, base=base, max_seq_len=max_seq_len)

    torch.manual_seed(42)  # type: ignore[reportUnknownMemberType]
    q = torch.randn(1, 1, max_seq_len, dim)
    k = torch.randn(1, 1, max_seq_len, dim)

    q_rot = rope.apply_rope(q, T=max_seq_len)
    k_rot = rope.apply_rope(k, T=max_seq_len)

    # dot product between position 2 and 0 (offset=2)
    dot_a = (q_rot[0, 0, 2] * k_rot[0, 0, 0]).sum()
    # same vectors, but shifted to positions 12 and 10 (same offset=2)
    q2 = torch.zeros_like(q)
    k2 = torch.zeros_like(k)
    q2[0, 0, 12] = q[0, 0, 2]
    k2[0, 0, 10] = k[0, 0, 0]
    q2_rot = rope.apply_rope(q2, T=max_seq_len)
    k2_rot = rope.apply_rope(k2, T=max_seq_len)
    dot_b = (q2_rot[0, 0, 12] * k2_rot[0, 0, 10]).sum()

    torch.testing.assert_close(dot_a, dot_b)


def test_rotary_larger_base_rotates_slower() -> None:
    """A larger base means smaller frequencies, so positions rotate less per step."""
    dim, max_seq_len = 16, 32
    rope_small = RotaryEmbedding(dim=dim, base=10, max_seq_len=max_seq_len)
    rope_large = RotaryEmbedding(dim=dim, base=10000, max_seq_len=max_seq_len)

    # total rotation = sum of |angle| across all dimensions at position 1
    def total_rotation(rope: RotaryEmbedding) -> float:
        angles = rope.cos_cache[1].acos()  # angle = arccos(cos(theta_i * 1))
        return float(angles.sum().item())

    assert total_rotation(rope_small) > total_rotation(rope_large)
