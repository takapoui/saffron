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

    def decode(self, tokens: list[int]) -> str:
        # Predictable: just hyphen-join the ids. Lets tests assert on content
        # without depending on a real vocab.
        return "-".join(str(t) for t in tokens)


class _FakeModel:
    """Returns a preset `generated` tensor; records the args generate received."""

    def __init__(self, generated: torch.Tensor, stop_ids: list[int]) -> None:
        self._generated = generated
        self._tokenizer = _FakeTokenizer(stop_ids)
        self.last_idx: torch.Tensor | None = None
        self.last_attention_mask: torch.Tensor | None = None

    def get_tokenizer(self) -> _FakeTokenizer:
        return self._tokenizer

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        stop_token_ids: list[int],
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
    )
    # The fake model returned a fixed tensor; rb.input_ids must equal it.
    assert torch.equal(rb.input_ids, model._generated)  # type: ignore[reportPrivateUsage]
