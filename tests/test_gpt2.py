import pytest
import torch

from saffron.models import GPT2, GPT2Config


@pytest.fixture
def config() -> GPT2Config:
    return GPT2Config(
        vocab_size=1000,
        n_embd=96,
        block_size=64,
        n_layer=12,
        n_head=12,
    )


def test_forward_shape(config: GPT2Config) -> None:
    model = GPT2(config)
    idx = torch.randint(0, config.vocab_size, (2, 10))  # batch=2, seq_len=10
    logits, _ = model(idx)
    assert logits.shape == (2, 10, config.vocab_size)  # (batch, seq_len, vocab_size)
