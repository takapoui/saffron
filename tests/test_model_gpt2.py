"""Tests for the GPT-2 model (forward, generate, per-row stop)."""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest
import torch

from saffron.constants import LABEL_IGNORE_INDEX
from saffron.model import GPT2, GPT2Config


def _fake_tokenizer(stop_ids: list[int]) -> MagicMock:
    """Tokenizer mock with the two properties generate reads off."""
    tok = MagicMock()
    tok.stop_token_ids = stop_ids
    tok.pad_token_id = stop_ids[0]
    return tok


@pytest.fixture
def config() -> GPT2Config:
    return GPT2Config(
        vocab_size=1000,
        n_embd=96,
        block_size=64,
        n_layer=2,
        n_head=4,
    )


@pytest.fixture
def model(config: GPT2Config) -> GPT2:
    torch.manual_seed(0)  # type: ignore[reportUnknownMemberType]
    return GPT2(config)


# --- forward ---


def test_forward_shape(model: GPT2, config: GPT2Config) -> None:
    idx = torch.randint(0, config.vocab_size, (2, 10))
    logits, loss = model(idx)
    assert logits.shape == (2, 10, config.vocab_size)
    assert loss is None


def test_forward_with_targets_returns_scalar_loss(model: GPT2, config: GPT2Config) -> None:
    idx = torch.randint(0, config.vocab_size, (2, 10))
    target = torch.randint(0, config.vocab_size, (2, 10))
    _, loss = model(idx, target)
    assert loss is not None
    assert loss.shape == ()
    assert loss.item() > 0


def test_forward_loss_ignores_label_ignore_index(model: GPT2, config: GPT2Config) -> None:
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


def test_attention_mask_ignores_padding(model: GPT2, config: GPT2Config) -> None:
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


def test_attention_mask_wrong_shape_raises(model: GPT2, config: GPT2Config) -> None:
    idx = torch.randint(0, config.vocab_size, (2, 10))
    bad_mask = torch.ones(2, 10, 10)  # should be (B, T)
    with pytest.raises(AssertionError):
        model(idx, attention_mask=bad_mask)


# --- generate ---


def test_generate_output_length(model: GPT2, config: GPT2Config) -> None:
    idx = torch.randint(0, config.vocab_size, (1, 5))
    out = model.generate(idx, max_new_tokens=10)
    assert out.shape == (1, 15)  # 5 prompt + 10 generated


def test_generate_stops_at_stop_token(model: GPT2, config: GPT2Config) -> None:
    stop_token = 0
    model._tokenizer = _fake_tokenizer([stop_token])  # pyright: ignore[reportPrivateUsage]
    idx = torch.randint(1, config.vocab_size, (1, 5))  # no stop tokens in prompt
    out = model.generate(idx, max_new_tokens=50)
    # output should be <= 5 + 50
    assert out.shape[1] <= 55


def test_generate_batch(model: GPT2, config: GPT2Config) -> None:
    idx = torch.randint(0, config.vocab_size, (3, 5))
    out = model.generate(idx, max_new_tokens=8)
    assert out.shape == (3, 13)


def test_generate_top_p_raises_for_native_models(model: GPT2, config: GPT2Config) -> None:
    idx = torch.randint(0, config.vocab_size, (1, 5))
    with pytest.raises(ValueError, match="top_p"):
        model.generate(idx, max_new_tokens=8, top_p=0.9)


# --- base_model.generate: per-row stop ---


def test_base_model_generate_stops_finished_rows_only(config: GPT2Config) -> None:
    """When one row in a batch hits a stop token, subsequent tokens for that
    row must be clamped to the stop token ID while generation continues for
    rows that have not yet finished."""
    STOP = 5  # token id used as stop; must be < config.vocab_size

    class _DetGPT2(GPT2):
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
    model = _DetGPT2(config)
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
