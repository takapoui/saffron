"""Tests for compute_token_log_probs (per-token log p(next_token | prefix))."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from saffron.rl.logprobs import compute_token_log_probs


class _FakeModel:
    """Minimal stand-in for BaseModel. Returns preset logits, records call args.

    Avoids loading real weights — we only care that compute_token_log_probs
    forwards inputs correctly and post-processes logits correctly.
    """

    def __init__(self, logits: torch.Tensor) -> None:
        self._logits = logits
        self.last_idx: torch.Tensor | None = None
        self.last_attention_mask: torch.Tensor | None = None

    def __call__(
        self,
        idx: torch.Tensor,
        target: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self.last_idx = idx
        self.last_attention_mask = attention_mask
        return self._logits, None


def test_output_shape_is_t_minus_1() -> None:
    """Output drops one position vs input (no successor for the last token)."""
    B, T, V = 2, 5, 7
    logits = torch.randn(B, T - 1, V)
    model = _FakeModel(logits)
    input_ids = torch.randint(0, V, (B, T))
    attention_mask = torch.ones((B, T), dtype=torch.long)

    out = compute_token_log_probs(model, input_ids, attention_mask, temperature=1.0)  # type: ignore[arg-type]

    assert out.shape == (B, T - 1)


def test_input_slicing_and_mask_passthrough() -> None:
    """Model receives input_ids[:, :-1] and attention_mask[:, :-1]."""
    B, T, V = 1, 4, 5
    logits = torch.randn(B, T - 1, V)
    model = _FakeModel(logits)
    input_ids = torch.tensor([[0, 1, 2, 3]])
    attention_mask = torch.tensor([[1, 1, 0, 0]])

    compute_token_log_probs(model, input_ids, attention_mask, temperature=1.0)  # type: ignore[arg-type]

    assert model.last_idx is not None and model.last_attention_mask is not None
    assert torch.equal(model.last_idx, torch.tensor([[0, 1, 2]]))
    assert torch.equal(model.last_attention_mask, torch.tensor([[1, 1, 0]]))


def test_gathers_log_prob_of_actual_next_token() -> None:
    """Each output position equals log_softmax(logits)[target_token] for that position."""
    B, T, V = 1, 3, 4
    # Hand-crafted logits, two positions (T-1 = 2):
    logits = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0],  # position 0 predicts next token
                [0.0, 0.0, 0.0, 0.0],  # position 1 predicts next token (uniform)
            ]
        ]
    )
    model = _FakeModel(logits)
    # input_ids[:, 1:] = [[2, 0]] — targets at positions 0 and 1
    input_ids = torch.tensor([[9, 2, 0]])
    attention_mask = torch.ones((B, T), dtype=torch.long)

    out = compute_token_log_probs(model, input_ids, attention_mask, temperature=1.0)  # type: ignore[arg-type]

    expected = F.log_softmax(logits, dim=-1)
    # position 0 gathered at token 2; position 1 gathered at token 0 (uniform → log(1/V))
    assert out[0, 0].item() == pytest.approx(expected[0, 0, 2].item())  # type: ignore[reportUnknownMemberType]
    assert out[0, 1].item() == pytest.approx(math.log(1.0 / V))  # type: ignore[reportUnknownMemberType]


def test_temperature_scales_logits_before_log_softmax() -> None:
    """temperature=T means log_probs come from softmax(logits/T)."""
    B = 1
    logits = torch.tensor([[[2.0, 0.0, -2.0]]])  # (1, 1, 3)
    model_t1 = _FakeModel(logits.clone())
    model_t2 = _FakeModel(logits.clone())
    input_ids = torch.tensor([[5, 0]])  # target = token 0
    attention_mask = torch.ones((B, 2), dtype=torch.long)

    out_t1 = compute_token_log_probs(model_t1, input_ids, attention_mask, temperature=1.0)  # type: ignore[arg-type]
    out_t2 = compute_token_log_probs(model_t2, input_ids, attention_mask, temperature=2.0)  # type: ignore[arg-type]

    expected_t1 = F.log_softmax(logits, dim=-1)[0, 0, 0]
    expected_t2 = F.log_softmax(logits / 2.0, dim=-1)[0, 0, 0]
    assert out_t1[0, 0].item() == pytest.approx(expected_t1.item())  # type: ignore[reportUnknownMemberType]
    assert out_t2[0, 0].item() == pytest.approx(expected_t2.item())  # type: ignore[reportUnknownMemberType]
    # Higher temperature → flatter distribution → log-prob of any single token moves toward log(1/V)
    assert out_t2[0, 0].item() != pytest.approx(out_t1[0, 0].item())  # type: ignore[reportUnknownMemberType]


def test_log_probs_are_non_positive() -> None:
    """log_softmax outputs are always ≤ 0."""
    B, T, V = 3, 6, 10
    logits = torch.randn(B, T - 1, V) * 5.0
    model = _FakeModel(logits)
    input_ids = torch.randint(0, V, (B, T))
    attention_mask = torch.ones((B, T), dtype=torch.long)

    out = compute_token_log_probs(model, input_ids, attention_mask, temperature=1.3)  # type: ignore[arg-type]

    assert torch.all(out <= 0)
