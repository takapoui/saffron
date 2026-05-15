"""Tests for TiktokenTokenizer.apply_chat_template()."""

from __future__ import annotations

import pytest

from saffron.tokenizer import TiktokenTokenizer


@pytest.fixture
def tok() -> TiktokenTokenizer:
    return TiktokenTokenizer("gpt2")


def test_user_only(tok: TiktokenTokenizer) -> None:
    tokens = tok.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        add_generation_prompt=False,
    )
    text = tok.decode(tokens)
    assert "<|user|>" in text
    assert "hello" in text


def test_system_user(tok: TiktokenTokenizer) -> None:
    tokens = tok.apply_chat_template(
        [{"role": "system", "content": "you are helpful"}, {"role": "user", "content": "hi"}],
        add_generation_prompt=False,
    )
    text = tok.decode(tokens)
    assert "<|system|>" in text
    assert "<|user|>" in text


def test_assistant_appends_eot(tok: TiktokenTokenizer) -> None:
    tokens = tok.apply_chat_template(
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        add_generation_prompt=False,
    )
    assert tokens[-1] == tok.eot_token


def test_add_generation_prompt(tok: TiktokenTokenizer) -> None:
    without = tok.apply_chat_template(
        [{"role": "user", "content": "q"}],
        add_generation_prompt=False,
    )
    with_prompt = tok.apply_chat_template(
        [{"role": "user", "content": "q"}],
        add_generation_prompt=True,
    )
    assert len(with_prompt) > len(without)
    assert tok.decode(with_prompt).endswith("<|assistant|>\n")


def test_unknown_role_raises(tok: TiktokenTokenizer) -> None:
    with pytest.raises(ValueError, match="Unknown role"):
        tok.apply_chat_template(
            [{"role": "banana", "content": "oops"}],
            add_generation_prompt=False,
        )
