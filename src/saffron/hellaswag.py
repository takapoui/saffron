import contextlib
import logging
from typing import Any, cast

import tiktoken
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset  # type: ignore[reportUnknownVariableType]

from .models import BaseModel

logger = logging.getLogger(__name__)

_hellaswag_dataset: Dataset | None = None


def _get_hellaswag_dataset() -> Dataset:
    global _hellaswag_dataset
    if _hellaswag_dataset is None:
        _hellaswag_dataset = load_dataset("hellaswag", split="validation", trust_remote_code=True)
        logger.info("Downloaded hellaswag dataset.")
    return _hellaswag_dataset


def tokenize_example(
    example: dict[str, Any], enc: tiktoken.Encoding
) -> tuple[torch.Tensor, torch.Tensor]:
    ctx_token: list[int] = enc.encode_ordinary(example["ctx"])
    ending_tokens_list: list[list[int]] = []
    for ending in example["endings"]:
        ending_tokens_list.append(enc.encode_ordinary(" " + ending))
    tensor_length = len(ctx_token) + max(len(seq) for seq in ending_tokens_list)
    tokens = torch.zeros((4, tensor_length), dtype=torch.long)
    tokens[:, : len(ctx_token)] = torch.tensor(ctx_token, dtype=torch.long)
    for idx, row in enumerate(ending_tokens_list):
        tokens[idx, len(ctx_token) : len(ctx_token) + len(row)] = torch.tensor(
            row, dtype=torch.long
        )
    x, y = tokens[:, :-1].clone(), tokens[:, 1:].clone()
    y[:, : len(ctx_token) - 1] = -1
    for idx, row in enumerate(ending_tokens_list):
        y[idx, len(ctx_token) + len(row) - 1 :] = -1
    return x, y


@torch.no_grad()
def evaluate_hellaswag(
    model: BaseModel, device: str, device_type: str, enc: tiktoken.Encoding
) -> float:
    hellaswag = _get_hellaswag_dataset()

    model.eval()
    example_count = 0
    correct_count = 0
    for example in hellaswag:  # type: ignore[reportUnknownVariableType]
        example = cast(dict[str, Any], example)  # poor stubs
        x, y = tokenize_example(example, enc=enc)
        # truncate if too long
        x, y = (
            x[:, : model.config.block_size].to(device),
            y[:, : model.config.block_size].to(device),
        )
        # annoyance: bfloat16 autocast is unstable on MPS with varying batch size,
        # we run in float32 instead to avoid crashing.
        ctx = (
            torch.autocast(device_type=device_type, dtype=torch.bfloat16)
            if device_type == "cuda"
            else contextlib.nullcontext()
        )
        with ctx:
            logits, _ = model(x)
            loss = (
                F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    y.reshape(-1),
                    reduction="none",
                    ignore_index=-1,
                )
                .reshape(4, -1)
                .sum(dim=1)
                / (y != -1).sum(dim=1).float()
            )
            guess = int(loss.argmin().item())

        example_count += 1
        correct_count += guess == int(example["label"])

    model.train()
    return correct_count / example_count
