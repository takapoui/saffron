"""Tests for rollout (sample G completions per prompt, build masks)."""

from __future__ import annotations

import torch

from saffron.rl.rollout import RolloutBatch, rollout


class _FakeTokenizer:
    """Minimal tokenizer stand-in: fixed stop_token_ids, predictable decode."""

    def __init__(self, stop_ids: list[int]) -> None:
        self._stop_ids = stop_ids

    @property
    def stop_token_ids(self) -> list[int]:
        return self._stop_ids

    @property
    def pad_token_id(self) -> int:
        return self._stop_ids[0]

    def decode(self, tokens: list[int]) -> str:
        # Predictable: just hyphen-join the ids. Lets tests assert on content
        # without depending on a real vocab.
        return "-".join(str(t) for t in tokens)


class _FakeModel:
    """Returns a preset `generated` tensor; records the args generate received."""

    def __init__(self, generated: torch.Tensor, stop_ids: list[int]) -> None:
        self._generated = generated
        self._fake_tokenizer = _FakeTokenizer(stop_ids)
        self.last_idx: torch.Tensor | None = None
        self.last_attention_mask: torch.Tensor | None = None

    @property
    def tokenizer(self) -> _FakeTokenizer:
        return self._fake_tokenizer

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        self.last_idx = idx
        self.last_attention_mask = attention_mask
        return self._generated


def _make_running_example() -> tuple[_FakeModel, torch.Tensor, torch.Tensor]:
    """The same 2-prompt × 3-sample example we worked through in design.

    T_prompt=4, max_new_tokens=5, T_full=9, pad_id=9.
    Completions chosen to exercise: EOS-early, no-EOS, EOS-very-early.
    """
    prompt_ids = torch.tensor(
        [
            [10, 11, 12, 13],
            [10, 11, 14, 15],
        ]
    )
    prompt_attn = torch.ones_like(prompt_ids)

    # After repeat_interleave(3) the rollout will get prompts in this order:
    #   rows 0,1,2: prompt 0 (=[10,11,12,13])
    #   rows 3,4,5: prompt 1 (=[10,11,14,15])
    # Then "generate" fills positions 4..8.
    generated = torch.tensor(
        [
            [10, 11, 12, 13, 20, 21, 9, 9, 9],  # finished at pos 6, padded
            [10, 11, 12, 13, 22, 23, 24, 25, 26],  # ran out of budget (no EOS)
            [10, 11, 12, 13, 27, 9, 9, 9, 9],  # finished at pos 5
            [10, 11, 14, 15, 30, 31, 32, 9, 9],  # finished at pos 7
            [10, 11, 14, 15, 33, 34, 35, 36, 37],  # no EOS
            [10, 11, 14, 15, 38, 9, 9, 9, 9],  # finished at pos 5
        ]
    )
    model = _FakeModel(generated, stop_ids=[9])
    return model, prompt_ids, prompt_attn


def test_returns_rollout_batch_with_expected_fields() -> None:
    model, prompt_ids, prompt_attn = _make_running_example()

    rb = rollout(
        model=model,  # type: ignore[arg-type]
        prompt_input_ids=prompt_ids,
        prompt_attention_mask=prompt_attn,
        group_size=3,
        max_new_tokens=5,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
    )

    assert isinstance(rb, RolloutBatch)
    assert rb.input_ids.shape == (6, 9)
    assert rb.attention_mask.shape == (6, 9)
    assert rb.response_mask.shape == (6, 9)
    assert len(rb.completion_texts) == 6
    assert len(rb.response_lens) == 6


def test_repeat_interleave_layout_passed_to_generate() -> None:
    """Rows 0..G-1 should be copies of prompt 0; rows G..2G-1 copies of prompt 1."""
    model, prompt_ids, prompt_attn = _make_running_example()

    rollout(
        model=model,  # type: ignore[arg-type]
        prompt_input_ids=prompt_ids,
        prompt_attention_mask=prompt_attn,
        group_size=3,
        max_new_tokens=5,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
    )

    assert model.last_idx is not None
    expected_idx = torch.tensor(
        [
            [10, 11, 12, 13],
            [10, 11, 12, 13],
            [10, 11, 12, 13],
            [10, 11, 14, 15],
            [10, 11, 14, 15],
            [10, 11, 14, 15],
        ]
    )
    assert torch.equal(model.last_idx, expected_idx)


def test_response_lens_match_first_stop_position() -> None:
    """Each row's response_len = position of first stop in completion, or full length if no stop."""
    model, prompt_ids, prompt_attn = _make_running_example()
    rb = rollout(
        model=model,  # type: ignore[arg-type]
        prompt_input_ids=prompt_ids,
        prompt_attention_mask=prompt_attn,
        group_size=3,
        max_new_tokens=5,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
    )
    # Expected from the running example (positions of first 9 in completion,
    # +1 to include EOS itself; 5 if no EOS).
    assert rb.response_lens == [3, 5, 2, 4, 5, 2]


def test_response_mask_matches_response_lens_via_sum() -> None:
    """response_mask.sum(-1) and response_lens must agree (consistency invariant)."""
    model, prompt_ids, prompt_attn = _make_running_example()
    rb = rollout(
        model=model,  # type: ignore[arg-type]
        prompt_input_ids=prompt_ids,
        prompt_attention_mask=prompt_attn,
        group_size=3,
        max_new_tokens=5,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
    )
    assert rb.response_mask.sum(dim=-1).tolist() == rb.response_lens  # pyright: ignore[reportUnknownMemberType]


def test_response_mask_zero_on_prompt_positions() -> None:
    """response_mask must be 0 for the first T_prompt columns of every row."""
    model, prompt_ids, prompt_attn = _make_running_example()
    rb = rollout(
        model=model,  # type: ignore[arg-type]
        prompt_input_ids=prompt_ids,
        prompt_attention_mask=prompt_attn,
        group_size=3,
        max_new_tokens=5,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
    )
    assert torch.all(rb.response_mask[:, :4] == 0)


def test_attention_mask_includes_eos_excludes_padding() -> None:
    """attention_mask is 1 through the EOS token, 0 strictly after."""
    model, prompt_ids, prompt_attn = _make_running_example()
    rb = rollout(
        model=model,  # type: ignore[arg-type]
        prompt_input_ids=prompt_ids,
        prompt_attention_mask=prompt_attn,
        group_size=3,
        max_new_tokens=5,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
    )
    expected = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 0, 0],  # EOS at pos 6
            [1, 1, 1, 1, 1, 1, 1, 1, 1],  # no EOS
            [1, 1, 1, 1, 1, 1, 0, 0, 0],  # EOS at pos 5
            [1, 1, 1, 1, 1, 1, 1, 1, 0],  # EOS at pos 7
            [1, 1, 1, 1, 1, 1, 1, 1, 1],  # no EOS
            [1, 1, 1, 1, 1, 1, 0, 0, 0],  # EOS at pos 5
        ]
    )
    assert torch.equal(rb.attention_mask, expected)


def test_completion_texts_exclude_stop_tokens() -> None:
    """Decode strips stop tokens; EOS-finished rows don't include '9' in their text."""
    model, prompt_ids, prompt_attn = _make_running_example()
    rb = rollout(
        model=model,  # type: ignore[arg-type]
        prompt_input_ids=prompt_ids,
        prompt_attention_mask=prompt_attn,
        group_size=3,
        max_new_tokens=5,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
    )
    # Our fake decode hyphen-joins ids; "9" appearing in text would mean
    # the stop token leaked through.
    for i, text in enumerate(rb.completion_texts):
        tokens_str = text.split("-") if text else []
        assert "9" not in tokens_str, f"row {i}: stop token leaked into {text!r}"


def test_completion_texts_have_correct_content() -> None:
    """Decoded content matches the completion slice with stop tokens removed."""
    model, prompt_ids, prompt_attn = _make_running_example()
    rb = rollout(
        model=model,  # type: ignore[arg-type]
        prompt_input_ids=prompt_ids,
        prompt_attention_mask=prompt_attn,
        group_size=3,
        max_new_tokens=5,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
    )
    # Built from the running example: completion tokens with stop (9) stripped,
    # hyphen-joined by our fake decoder.
    expected = [
        "20-21",  # row 0: [20,21,9] → strip 9
        "22-23-24-25-26",  # row 1: no EOS
        "27",  # row 2: [27,9] → strip 9
        "30-31-32",  # row 3: [30,31,32,9] → strip 9
        "33-34-35-36-37",  # row 4: no EOS
        "38",  # row 5: [38,9] → strip 9
    ]
    assert rb.completion_texts == expected


def test_boundary_is_first_stop_token_not_pad() -> None:
    """Model emits a non-primary stop token (e.g. <|im_end|>) while pad is the
    primary stop (e.g. <|endoftext|>). Boundary must be at the emitted stop,
    not at the first pad — otherwise the pad token itself is treated as content."""
    # stop_token_ids = [8, 9] → pad_token_id = 8 (primary), 9 is the other stop.
    # Row 0: emits 9 at position 5, then pads with 8.
    # Row 1: emits 8 (which happens to also be pad) at position 6.
    prompt_ids = torch.tensor([[10, 11, 12, 13], [10, 11, 12, 13]])
    prompt_attn = torch.ones_like(prompt_ids)
    generated = torch.tensor(
        [
            [10, 11, 12, 13, 20, 9, 8, 8, 8],  # emits 9 (secondary stop), pads with 8
            [10, 11, 12, 13, 30, 31, 8, 8, 8],  # emits 8 (primary stop = pad)
        ]
    )
    model = _FakeModel(generated, stop_ids=[8, 9])

    rb = rollout(
        model=model,  # type: ignore[arg-type]
        prompt_input_ids=prompt_ids,
        prompt_attention_mask=prompt_attn,
        group_size=1,
        max_new_tokens=5,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
    )

    # Row 0: position 5 (the 9) is the stop — valid through there, 0 after.
    # Row 1: position 6 (the first 8) is the stop — valid through there, 0 after.
    assert rb.response_lens == [2, 3]
    expected_response = torch.tensor(
        [
            [0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0],
        ]
    )
    assert torch.equal(rb.response_mask, expected_response)


def test_left_padded_prompts_preserve_mask_in_prefix() -> None:
    """Prompts with leading pad (left-padding for batched generation) must keep
    their mask intact in the prefix — we must not silently flip pads to 1s."""
    # Two prompts of different real lengths, left-padded to length 4.
    # Row 0: 2 pad tokens then real prompt. Row 1: full real prompt.
    prompt_ids = torch.tensor(
        [
            [99, 99, 12, 13],  # 99s are pad on the left
            [10, 11, 14, 15],
        ]
    )
    prompt_attn = torch.tensor(
        [
            [0, 0, 1, 1],
            [1, 1, 1, 1],
        ]
    )
    # Fake generated tensor: prompts unchanged, completion of length 3 (no EOS in completion).
    generated = torch.tensor(
        [
            [99, 99, 12, 13, 50, 51, 52],
            [10, 11, 14, 15, 60, 61, 62],
        ]
    )
    model = _FakeModel(generated, stop_ids=[9])

    rb = rollout(
        model=model,  # type: ignore[arg-type]
        prompt_input_ids=prompt_ids,
        prompt_attention_mask=prompt_attn,
        group_size=1,
        max_new_tokens=3,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
    )

    expected_attn = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1, 1],  # prefix preserves the 0s, completion all valid
            [1, 1, 1, 1, 1, 1, 1],
        ]
    )
    assert torch.equal(rb.attention_mask, expected_attn)


def test_input_ids_is_the_full_generated_sequence() -> None:
    """rb.input_ids is exactly what generate returned (prompt + completion + padding)."""
    model, prompt_ids, prompt_attn = _make_running_example()
    rb = rollout(
        model=model,  # type: ignore[arg-type]
        prompt_input_ids=prompt_ids,
        prompt_attention_mask=prompt_attn,
        group_size=3,
        max_new_tokens=5,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
    )
    # The fake model returned a fixed tensor; rb.input_ids must equal it.
    assert torch.equal(rb.input_ids, model._generated)  # type: ignore[reportPrivateUsage]
