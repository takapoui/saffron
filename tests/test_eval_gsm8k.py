"""Tests for extract_answer() and evaluate_gsm8k()."""

from __future__ import annotations

import pytest
import torch

from saffron.constants import LABEL_IGNORE_INDEX
from saffron.eval.config import EvalGSM8KConfig
from saffron.eval.gsm8k import evaluate_gsm8k, extract_answer

# ---------------------------------------------------------------------------
# Shared stubs (GSM8K evaluation needs a tokenizer, model, and val-loader)
# ---------------------------------------------------------------------------

_STOP = 9999
_IGN = LABEL_IGNORE_INDEX

# Maps specific token-id tuples → human-readable strings.
# Unknown tuples → "" (no parseable answer).
_DECODE_TABLE: dict[tuple[int, ...], str] = {
    (11, 12, 13): "The answer is #### 42",  # ex-1 ground truth
    (21, 22): "The answer is #### 7",  # ex-2 ground truth
    (31, 32): "#### 42",  # what stub model generates
    (51,): "no parseable marker here",  # deliberately un-parseable
    (61, 62): "#### 61",  # attention-mask test gt-A
    (71,): "#### 71",  # attention-mask test gt-B
}


class _StubTok:
    name = "stub"
    stop_token_ids: list[int] = [_STOP]
    pad_token_id: int = _STOP

    def decode(self, ids: list[int]) -> str:
        return _DECODE_TABLE.get(tuple(ids), "")

    def encode(self, text: str) -> list[int]:
        return [1]


class _StubModel:
    """Always appends [31, 32, STOP] to the prompt — decodes to '#### 42'."""

    @property
    def tokenizer(self) -> _StubTok:
        return _StubTok()

    def eval(self) -> None:
        pass

    def train(self) -> None:
        pass

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 50,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = idx.shape[0]
        suffix = torch.tensor([[31, 32, _STOP]], dtype=torch.long).repeat(B, 1)
        return torch.cat([idx, suffix], dim=1)


class _FakeValLoader:
    """Minimal SFTDataLoader stand-in accepted by evaluate_gsm8k."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self._x = x
        self._y = y

    @property
    def n_steps(self) -> int:
        return 1

    def reset(self) -> None:
        pass

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._x, self._y


def _two_example_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """
    x[0]: prompt=[1,2,3], answer=[11,12,13] → gt=42
    x[1]: prompt=[1,2,3], answer=[21,22]    → gt=7
    Stub model always generates [31,32] → "#### 42": ex-0 correct, ex-1 wrong.
    Expected accuracy = 0.5.
    """
    x = torch.tensor(
        [
            [1, 2, 3, 11, 12, 13, _STOP, 0],
            [1, 2, 3, 21, 22, _STOP, 0, 0],
        ],
        dtype=torch.long,
    )
    y = torch.tensor(
        [
            [_IGN, _IGN, 11, 12, 13, _IGN, _IGN, _IGN],
            [_IGN, _IGN, 21, 22, _IGN, _IGN, _IGN, _IGN],
        ],
        dtype=torch.long,
    )
    return x, y


# ---------------------------------------------------------------------------
# extract_answer
# ---------------------------------------------------------------------------


def test_normal() -> None:
    assert extract_answer("So the answer is #### 42") == 42.0


def test_negative() -> None:
    assert extract_answer("#### -5") == -5.0


def test_with_commas() -> None:
    assert extract_answer("#### 1,234") == 1234.0


def test_decimal() -> None:
    assert extract_answer("#### 3.14") == 3.14


def test_no_marker() -> None:
    assert extract_answer("The answer is 42") is None


def test_empty() -> None:
    assert extract_answer("") is None


def test_junk_after_marker() -> None:
    # only the first match matters
    assert extract_answer("#### 10 apples left") == 10.0


def test_mid_text() -> None:
    assert extract_answer("We get #### 7\nSome trailing text") == 7.0


# ---------------------------------------------------------------------------
# evaluate_gsm8k
# ---------------------------------------------------------------------------


def test_evaluate_gsm8k_scores_controlled_examples() -> None:
    x, y = _two_example_batch()
    acc = evaluate_gsm8k(
        model=_StubModel(),  # type: ignore[arg-type]
        val_loader=_FakeValLoader(x, y),  # type: ignore[arg-type]
        device="cpu",
        device_type="cpu",
        config=EvalGSM8KConfig(every=None, max_tokens=5, gen_batch_size=8),
    )
    assert acc == pytest.approx(0.5)  # type: ignore[reportUnknownMemberType]


def test_evaluate_gsm8k_ignores_examples_without_parseable_answer() -> None:
    """An example whose answer portion has no '####' marker must be skipped,
    not counted as wrong."""
    x = torch.tensor(
        [
            [1, 2, 3, 11, 12, 13, _STOP, 0],  # parseable → gt=42
            [1, 2, 3, 51, _STOP, 0, 0, 0],  # decode([51]) has no #### → skip
        ],
        dtype=torch.long,
    )
    y = torch.tensor(
        [
            [_IGN, _IGN, 11, 12, 13, _IGN, _IGN, _IGN],
            [_IGN, _IGN, 51, _IGN, _IGN, _IGN, _IGN, _IGN],
        ],
        dtype=torch.long,
    )
    # Only 1 valid example; stub generates "#### 42" → correct.
    acc = evaluate_gsm8k(
        model=_StubModel(),  # type: ignore[arg-type]
        val_loader=_FakeValLoader(x, y),  # type: ignore[arg-type]
        device="cpu",
        device_type="cpu",
        config=EvalGSM8KConfig(every=None, max_tokens=5, gen_batch_size=8),
    )
    assert acc == pytest.approx(1.0)  # type: ignore[reportUnknownMemberType]


def test_evaluate_gsm8k_uses_attention_mask_for_padded_batches() -> None:
    """When prompts have different lengths they are left-padded; the resulting
    attention_mask must have 0 for pad positions and 1 for real tokens."""
    # ex-A: prompt_len=2, ex-B: prompt_len=4  → max_len=4
    x = torch.tensor(
        [
            [10, 11, 61, 62, _STOP, 0, 0, 0],  # k=1 → prompt=[10,11]
            [20, 21, 22, 23, 71, _STOP, 0, 0],  # k=3 → prompt=[20,21,22,23]
        ],
        dtype=torch.long,
    )
    y = torch.tensor(
        [
            [_IGN, 61, 62, _IGN, _IGN, _IGN, _IGN, _IGN],
            [_IGN, _IGN, _IGN, 71, _IGN, _IGN, _IGN, _IGN],
        ],
        dtype=torch.long,
    )

    captured: dict[str, torch.Tensor] = {}

    class _SpyModel(_StubModel):
        def generate(
            self,
            idx: torch.Tensor,
            max_new_tokens: int,
            temperature: float = 1.0,
            top_k: int = 50,
            attention_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if attention_mask is not None:
                captured["attention_mask"] = attention_mask.clone()
            B = idx.shape[0]
            suffix = torch.zeros((B, max_new_tokens), dtype=torch.long)
            return torch.cat([idx, suffix], dim=1)

    evaluate_gsm8k(
        model=_SpyModel(),  # type: ignore[arg-type]
        val_loader=_FakeValLoader(x, y),  # type: ignore[arg-type]
        device="cpu",
        device_type="cpu",
        config=EvalGSM8KConfig(every=None, max_tokens=3, gen_batch_size=8),
    )

    mask = captured["attention_mask"]
    assert mask.shape == (2, 4)
    # shorter prompt: padded on the left with two zeros
    assert mask[0].tolist() == [0, 0, 1, 1]  # type: ignore[reportUnknownMemberType]
    # longer prompt: fully attended
    assert mask[1].tolist() == [1, 1, 1, 1]  # type: ignore[reportUnknownMemberType]
