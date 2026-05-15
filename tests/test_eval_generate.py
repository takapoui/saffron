"""Tests for evaluate_generate()."""

from __future__ import annotations

import torch

from saffron.eval.config import EvalGenerateConfig
from saffron.eval.generate import evaluate_generate
from saffron.tokenizer import HFTokenizer

_STOP = 9999


class _StubTok:
    name = "stub"
    stop_token_ids: list[int] = [_STOP]

    def decode(self, ids: list[int]) -> str:
        return " ".join(str(i) for i in ids)

    def encode(self, text: str) -> list[int]:
        return [10, 20, 30]  # fixed 3-token prompt


def test_evaluate_generate_stops_at_first_stop_token() -> None:
    """The completion must end before the first stop token; tokens after it
    must not appear in the decoded string."""

    class _SeqModel:
        def get_tokenizer(self) -> _StubTok:
            return _StubTok()

        def eval(self) -> None:
            pass

        def train(self) -> None:
            pass

        def generate(
            self,
            idx: torch.Tensor,
            max_new_tokens: int,
            temperature: float = 1.0,
            top_k: int = 50,
            stop_token_ids: list[int] | None = None,
            attention_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            B = idx.shape[0]
            # After prompt: token 100, STOP, token 200
            suffix = torch.tensor([[100, _STOP, 200]], dtype=torch.long).repeat(B, 1)
            return torch.cat([idx, suffix], dim=1)

    completions = evaluate_generate(
        model=_SeqModel(),  # type: ignore[arg-type]
        device="cpu",
        config=EvalGenerateConfig(
            every=None,
            prompt="hi",
            samples=1,
            max_tokens=3,
            use_chat_template=False,
        ),
    )

    assert len(completions) == 1
    # Token 200 (after the stop) must be absent; stop token itself excluded too
    assert str(_STOP) not in completions[0]
    assert "200" not in completions[0]
    # Token 100 (before stop) must be present
    assert "100" in completions[0]


def test_evaluate_generate_chat_template_path() -> None:
    """When use_chat_template=True the tokenizer's apply_chat_template must
    be called with the correct message structure."""

    class _FakeHFTok(HFTokenizer):
        """HFTokenizer subclass that skips network loading."""

        def __init__(self) -> None:  # type: ignore[override]
            self._name = "fake-hf"
            self.apply_calls: list[list[dict[str, str]]] = []

        @property
        def eot_token(self) -> int:
            return 0

        @property
        def stop_token_ids(self) -> list[int]:
            return [0]

        def encode(self, text: str) -> list[int]:
            return [1, 2]

        def decode(self, tokens: list[int]) -> str:
            return ""

        @property
        def vocab_size(self) -> int:
            return 100

        def apply_chat_template(  # type: ignore[override]
            self,
            messages: list[dict[str, str]],
            add_generation_prompt: bool,
            continue_final_message: bool = False,
        ) -> list[int]:
            self.apply_calls.append(messages)
            return [10, 11, 12]

    tok = _FakeHFTok()

    class _ChatModel:
        def get_tokenizer(self) -> _FakeHFTok:
            return tok

        def eval(self) -> None:
            pass

        def train(self) -> None:
            pass

        def generate(
            self,
            idx: torch.Tensor,
            max_new_tokens: int,
            temperature: float = 1.0,
            top_k: int = 50,
            stop_token_ids: list[int] | None = None,
            attention_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            B = idx.shape[0]
            return torch.cat([idx, torch.zeros((B, max_new_tokens), dtype=torch.long)], dim=1)

    evaluate_generate(
        model=_ChatModel(),  # type: ignore[arg-type]
        device="cpu",
        config=EvalGenerateConfig(
            every=None,
            prompt="What is 2+2?",
            samples=1,
            max_tokens=1,
            use_chat_template=True,
        ),
    )

    assert len(tok.apply_calls) == 1
    assert tok.apply_calls[0] == [{"role": "user", "content": "What is 2+2?"}]
