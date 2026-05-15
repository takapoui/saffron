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

    @property
    def stop_token_ids(self) -> list[int]:
        return [self.eot_token]

    @abstractmethod
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool,
        continue_final_message: bool = False,
    ) -> list[int]:
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

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool,
        continue_final_message: bool = False,
    ) -> list[int]:
        if continue_final_message:
            raise NotImplementedError("continue_final_message is not supported for tiktoken")
        tokens: list[int] = []
        for message in messages:
            role, content = message["role"], message["content"]
            if role == "system":
                tokens += self.encode(f"<|system|>\n{content}\n")
            elif role == "user":
                tokens += self.encode(f"<|user|>\n{content}\n")
            elif role == "assistant":
                tokens += self.encode(f"<|assistant|>\n{content}")
                tokens += [self.eot_token]
            else:
                raise ValueError(f"Unknown role: {role!r}")
        if add_generation_prompt:
            tokens += self.encode("<|assistant|>\n")
        return tokens


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

    @property
    def im_end_token(self) -> int | None:
        """Token ID for <|im_end|>, used as turn-end stop token in chat templates."""
        result = self._tok.convert_tokens_to_ids("<|im_end|>")  # pyright: ignore[reportUnknownMemberType]
        return result if isinstance(result, int) else None

    @property
    def stop_token_ids(self) -> list[int]:
        if self.im_end_token is not None:
            return [self.eot_token, self.im_end_token]
        return [self.eot_token]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool,
        continue_final_message: bool = False,
    ) -> list[int]:
        return self._tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=False,
            continue_final_message=continue_final_message,
        )
