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
            **{k: d[k] for k in ("use_chat_template", "temperature", "top_k") if k in d},
        )


@dataclass
class EvalHellaswagConfig:
    every: int | None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalHellaswagConfig:
        return cls(every=d.get("every"))


@dataclass
class EvalGSM8KConfig:
    every: int | None
    max_tokens: int = 500
    evaluate_on: int = 50  # TODO: remove cap and evaluate on full val set

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalGSM8KConfig:
        return cls(
            every=d.get("every"),
            **{k: d[k] for k in ("max_tokens", "evaluate_on") if k in d},
        )
