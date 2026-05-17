"""Tests for RLBatch.from_examples (left-padding, mask, samples ordering)."""

from __future__ import annotations

import torch

from saffron.data.rl.rl_dataloader import RLBatch


def _ex(input_ids: list[int], nums: list[int], target: int) -> dict[str, object]:
    return {"input_ids": input_ids, "nums": nums, "target": target}


def test_heterogeneous_lengths_are_left_padded() -> None:
    """Shorter prompts pad on the LEFT so all prompts share their right edge."""
    examples = [
        _ex(input_ids=[10, 11, 12, 13], nums=[1, 2], target=3),  # len 4
        _ex(input_ids=[20, 21], nums=[4, 5], target=9),  # len 2
        _ex(input_ids=[30, 31, 32], nums=[6, 7], target=13),  # len 3
    ]
    pad = 99

    batch = RLBatch.from_examples(examples, pad_token_id=pad)

    # Shape padded to longest (4)
    assert batch.input_ids.shape == (3, 4)
    expected_ids = torch.tensor(
        [
            [10, 11, 12, 13],  # no pad
            [99, 99, 20, 21],  # 2 left-pad
            [99, 30, 31, 32],  # 1 left-pad
        ]
    )
    assert torch.equal(batch.input_ids, expected_ids)


def test_attention_mask_matches_left_pad() -> None:
    """Attention mask is 0 on padded positions, 1 on real prompt positions."""
    examples = [
        _ex(input_ids=[10, 11, 12, 13], nums=[1], target=1),
        _ex(input_ids=[20, 21], nums=[2], target=2),
        _ex(input_ids=[30, 31, 32], nums=[3], target=3),
    ]
    batch = RLBatch.from_examples(examples, pad_token_id=99)

    expected_mask = torch.tensor(
        [
            [1, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
        ]
    )
    assert torch.equal(batch.attention_mask, expected_mask)


def test_samples_preserve_input_order() -> None:
    """`samples` list ordering matches the input examples; only nums/target carried."""
    examples = [
        _ex(input_ids=[1], nums=[1, 2, 3], target=6),
        _ex(input_ids=[2], nums=[4, 5], target=9),
        _ex(input_ids=[3], nums=[7], target=7),
    ]
    batch = RLBatch.from_examples(examples, pad_token_id=0)

    assert batch.samples == [
        {"nums": [1, 2, 3], "target": 6},
        {"nums": [4, 5], "target": 9},
        {"nums": [7], "target": 7},
    ]


def test_single_example_degenerate() -> None:
    """B=1 with a single prompt: no padding needed, mask all ones."""
    batch = RLBatch.from_examples(
        [_ex(input_ids=[10, 11, 12], nums=[1], target=1)],
        pad_token_id=0,
    )
    assert batch.input_ids.shape == (1, 3)
    assert torch.equal(batch.input_ids, torch.tensor([[10, 11, 12]]))
    assert torch.equal(batch.attention_mask, torch.tensor([[1, 1, 1]]))
    assert batch.samples == [{"nums": [1], "target": 1}]


def test_equal_length_examples_no_padding() -> None:
    """All same-length prompts → no pad cells, mask is all ones."""
    examples = [_ex([10, 11, 12], [1], 1), _ex([20, 21, 22], [2], 2)]
    batch = RLBatch.from_examples(examples, pad_token_id=99)

    assert torch.equal(batch.input_ids, torch.tensor([[10, 11, 12], [20, 21, 22]]))
    assert torch.equal(batch.attention_mask, torch.ones((2, 3), dtype=torch.long))


def test_pad_token_id_is_used() -> None:
    """The pad token id passed in is what actually fills the padded cells."""
    examples = [_ex([10], [1], 1), _ex([20, 21, 22], [2], 2)]
    pad = 12345

    batch = RLBatch.from_examples(examples, pad_token_id=pad)

    # Row 0: padded on the left with `pad`. Position 0,1 should be pad.
    assert batch.input_ids[0, 0].item() == pad
    assert batch.input_ids[0, 1].item() == pad
    # Real token preserved on the right
    assert batch.input_ids[0, 2].item() == 10


def test_dtypes() -> None:
    """input_ids and attention_mask are both long tensors."""
    batch = RLBatch.from_examples([_ex([10, 11], [1], 1)], pad_token_id=0)
    assert batch.input_ids.dtype == torch.long
    assert batch.attention_mask.dtype == torch.long
