"""Tests for VLLMTeacher._make_messages() — no vllm install required."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

# Stub vllm before saffron.model.vllm_teacher is imported so the module-level
# `from vllm import LLM, SamplingParams` doesn't fail when vllm isn't installed.
_vllm_stub = ModuleType("vllm")
_vllm_stub.LLM = MagicMock()  # type: ignore[attr-defined]
_vllm_stub.SamplingParams = MagicMock()  # type: ignore[attr-defined]
sys.modules.setdefault("vllm", _vllm_stub)

from saffron.model.vllm_teacher import VLLMTeacher, VLLMTeacherConfig  # noqa: E402


def _make_config(system_prompt: str | None) -> VLLMTeacherConfig:
    return VLLMTeacherConfig(
        model_name="dummy",
        max_tokens=128,
        temperature=0.7,
        dtype="bfloat16",
        system_prompt=system_prompt,
    )


@pytest.fixture
def teacher() -> VLLMTeacher:
    """Construct VLLMTeacher without calling __init__ (which loads a vLLM model)."""
    obj = VLLMTeacher.__new__(VLLMTeacher)
    obj.config = _make_config(system_prompt=None)
    obj._llm = MagicMock()  # type: ignore[reportPrivateUsage]
    obj._sampling_params = MagicMock()  # type: ignore[reportPrivateUsage]
    return obj


def test_make_messages_no_system_prompt(teacher: VLLMTeacher) -> None:
    msgs = teacher._make_messages("What is 2+2?")  # type: ignore[reportPrivateUsage]
    assert msgs == [{"role": "user", "content": "What is 2+2?"}]


def test_make_messages_with_system_prompt(teacher: VLLMTeacher) -> None:
    teacher.config = _make_config(system_prompt="You are a math tutor.")
    msgs = teacher._make_messages("What is 2+2?")  # type: ignore[reportPrivateUsage]
    assert msgs == [
        {"role": "system", "content": "You are a math tutor."},
        {"role": "user", "content": "What is 2+2?"},
    ]


def test_make_messages_system_prompt_comes_first(teacher: VLLMTeacher) -> None:
    teacher.config = _make_config(system_prompt="Be concise.")
    msgs = teacher._make_messages("hi")  # type: ignore[reportPrivateUsage]
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "user"
