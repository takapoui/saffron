from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from vllm import LLM, SamplingParams  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


@dataclass
class VLLMTeacherConfig:
    model_name: str
    max_tokens: int
    temperature: float
    dtype: str
    system_prompt: str | None


# We use vLLM for running larger models since it's faster than HF
class VLLMTeacher:
    # `Any` so pyright doesn't complain when vllm isn't installed locally
    _llm: Any
    _sampling_params: Any

    def __init__(self, config: VLLMTeacherConfig) -> None:
        self.config = config
        logger.info(f"Loading vLLM model: {config.model_name}")
        self._llm = LLM(model=config.model_name, dtype=config.dtype)
        self._sampling_params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    def generate(self, questions: list[str]) -> list[str]:
        message_batches = [self._make_messages(q) for q in questions]
        outputs = self._llm.chat(
            messages=message_batches,
            sampling_params=self._sampling_params,
        )
        # We sample n=1 per prompt, so each output has exactly one completion
        return [output.outputs[0].text for output in outputs]

    def _make_messages(self, question: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if self.config.system_prompt is not None:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": question})
        return messages
