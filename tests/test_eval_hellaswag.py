"""Tests for hellaswag.tokenize_example()."""

from __future__ import annotations

from saffron.eval.hellaswag import tokenize_example
from saffron.tokenizer import TiktokenTokenizer


def test_hellaswag_tokenize_example_masks_context_only() -> None:
    """Context positions in y must be -1; ending positions must carry real
    token ids; padding beyond each ending must be -1 again."""
    enc = TiktokenTokenizer("gpt2")
    ctx = "The quick brown fox"
    endings = ["jumped over the lazy dog", "sat down quietly", "flew away fast", "ran"]
    example: dict[str, object] = {"ctx": ctx, "endings": endings}

    x, y = tokenize_example(example, enc)  # type: ignore[arg-type]

    ctx_len = len(enc.encode(ctx))
    ending_lens = [len(enc.encode(" " + e)) for e in endings]

    assert x.shape[0] == 4, "should have one row per ending"

    # All rows: context positions (except last) must be masked
    assert (y[:, : ctx_len - 1] == -1).all().item()

    for i, el in enumerate(ending_lens):
        # The ending region (length el) must contain real token ids
        answer_region = y[i, ctx_len - 1 : ctx_len - 1 + el]
        assert (answer_region != -1).all().item(), f"row {i}: answer region has -1"
        # Positions after the ending must be padded with -1
        pad_region = y[i, ctx_len - 1 + el :]
        assert (pad_region == -1).all().item(), f"row {i}: unmasked padding"
