from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import tiktoken


class Tokenizer(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def eot_token(self) -> int:
        pass

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        pass

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        pass

    @staticmethod
    def from_name(name: str, local_files_only: bool = False) -> Tokenizer:
        if "/" in name:
            return HFTokenizer(name, local_files_only=local_files_only)
        return TiktokenTokenizer(name)


class TiktokenTokenizer(Tokenizer):
    def __init__(self, name: str) -> None:
        self._name = name
        self._enc = tiktoken.get_encoding(name)

    @property
    def name(self) -> str:
        return self._name

    @property
    def eot_token(self) -> int:
        return self._enc.eot_token

    def encode(self, text: str) -> list[int]:
        return self._enc.encode_ordinary(text)

    def decode(self, tokens: list[int]) -> str:
        return self._enc.decode(tokens)

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab


class HFTokenizer(Tokenizer):
    def __init__(self, name: str, local_files_only: bool = False) -> None:
        from transformers import AutoTokenizer  # type: ignore

        self._name = name
        self._tok: Any = AutoTokenizer.from_pretrained(name, local_files_only=local_files_only)  # pyright: ignore[reportUnknownMemberType]

    @property
    def name(self) -> str:
        return self._name

    @property
    def eot_token(self) -> int:
        return self._tok.eos_token_id

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, tokens: list[int]) -> str:
        return self._tok.decode(tokens)

    @property
    def vocab_size(self) -> int:
        return self._tok.vocab_size
