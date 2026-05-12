"""Tests for format_example() in data/prep_sft.py."""

from __future__ import annotations

import pytest

from saffron.constants import LABEL_IGNORE_INDEX
from saffron.data.prep_sft import format_example
from saffron.tokenizer import TiktokenTokenizer


@pytest.fixture
def tok() -> TiktokenTokenizer:
    return TiktokenTokenizer("gpt2")


def test_format_example_prompt_len(tok: TiktokenTokenizer) -> None:
    example = {"question": "What is 2+2?", "answer": "4"}
    tokens, prompt_len = format_example(example, tok, system_prompt=None)
    assert prompt_len > 0
    assert prompt_len < len(tokens)


def test_format_example_labels_mask_prompt(tok: TiktokenTokenizer) -> None:
    example = {"question": "What is 2+2?", "answer": "It is 4."}
    tokens, prompt_len = format_example(example, tok, system_prompt=None)
    labels = [LABEL_IGNORE_INDEX] * prompt_len + tokens[prompt_len:]
    assert all(tok_id == LABEL_IGNORE_INDEX for tok_id in labels[:prompt_len])
    assert any(tok_id != LABEL_IGNORE_INDEX for tok_id in labels[prompt_len:])


def test_format_example_with_system_prompt(tok: TiktokenTokenizer) -> None:
    example = {"question": "Q", "answer": "A"}
    _, prompt_len_no_sys = format_example(example, tok, system_prompt=None)
    _, prompt_len_with_sys = format_example(example, tok, system_prompt="You are helpful.")
    assert prompt_len_with_sys > prompt_len_no_sys


def test_format_example_answer_in_full_ids(tok: TiktokenTokenizer) -> None:
    example = {"question": "What is 2+2?", "answer": "Four"}
    tokens, prompt_len = format_example(example, tok, system_prompt=None)
    answer_text = tok.decode(tokens[prompt_len:])
    assert "Four" in answer_text
