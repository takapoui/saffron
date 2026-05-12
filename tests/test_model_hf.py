"""Tests for HFModel (model/hf.py)."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from saffron.model.hf import HFModel


def test_hf_generate_passes_stop_tokens_and_attention_mask() -> None:
    """HFModel.generate must forward input_ids, attention_mask, eos_token_id,
    and pad_token_id to the underlying hf_model.generate without alteration."""
    # Build a bare HFModel instance without touching AutoModelForCausalLM
    model: HFModel = HFModel.__new__(HFModel)
    model.hf_model = MagicMock()
    model.config = MagicMock()

    stop_ids = [50256, 50257]
    B, L = 2, 5
    input_ids = torch.randint(0, 1000, (B, L))
    attention_mask = torch.ones((B, L), dtype=torch.long)
    expected_out = torch.randint(0, 1000, (B, L + 3))
    model.hf_model.generate.return_value = expected_out  # type: ignore[union-attr]

    out = model.generate(
        idx=input_ids,
        max_new_tokens=3,
        temperature=1.0,
        top_k=50,
        stop_token_ids=stop_ids,
        attention_mask=attention_mask,
    )

    model.hf_model.generate.assert_called_once()  # type: ignore[union-attr]
    kw = model.hf_model.generate.call_args.kwargs  # type: ignore[union-attr]

    assert torch.equal(kw["input_ids"], input_ids)
    assert torch.equal(kw["attention_mask"], attention_mask)
    assert kw["eos_token_id"] == stop_ids
    assert kw["pad_token_id"] == stop_ids[0]
    assert out is expected_out
