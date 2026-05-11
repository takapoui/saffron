from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EvalLossConfig:
    every: int
    steps: int  # how many val batches to average over

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalLossConfig:
        return cls(every=d["every"], steps=d["steps"])


@dataclass
class EvalGenerateConfig:
    every: int | None
    prompt: str
    samples: int
    max_tokens: int
    use_chat_template: bool = False
    temperature: float = 1.0
    top_k: int = 50

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalGenerateConfig:
        return cls(
            every=d.get("every"),
            prompt=d["prompt"],
            samples=d["samples"],
            max_tokens=d["max_tokens"],
            use_chat_template=d.get("use_chat_template", False),
            temperature=d.get("temperature", 1.0),
            top_k=d.get("top_k", 50),
        )


@dataclass
class EvalHellaswagConfig:
    every: int | None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalHellaswagConfig:
        return cls(every=d.get("every"))
