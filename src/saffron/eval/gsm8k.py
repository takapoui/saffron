from __future__ import annotations

import contextlib
import logging
import re
from typing import TYPE_CHECKING, cast

import torch

from ..constants import LABEL_IGNORE_INDEX
from ..model.base_model import BaseModel

if TYPE_CHECKING:
    from ..data.sft_dataloader import SFTDataLoader

logger = logging.getLogger(__name__)

_ANSWER_RE = re.compile(r"####\s*([\d,.-]+)")


def extract_answer(text: str) -> float | None:
    match = _ANSWER_RE.search(text)
    if match is None:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


@torch.no_grad()
def evaluate_gsm8k(
    model: BaseModel,
    val_loader: SFTDataLoader,
    device: str,
    device_type: str,
    max_new_tokens: int,
    gen_batch_size: int,
) -> float:
    model.eval()
    tokenizer = model.get_tokenizer()

    stop_token_ids = tokenizer.stop_token_ids
    stop_set = set(stop_token_ids)
    pad_id = stop_token_ids[0]

    # Collect all (prompt, ground_truth) pairs from val set
    examples: list[tuple[torch.Tensor, float]] = []
    val_loader.reset()
    for _ in range(val_loader.n_steps):
        x, y = val_loader.next_batch()
        for b in range(x.shape[0]):
            y_b = y[b]
            answer_mask = y_b != LABEL_IGNORE_INDEX
            if not answer_mask.any():
                continue
            k = int(answer_mask.nonzero(as_tuple=False)[0].item())
            prompt_len = k + 1
            answer_toks = cast(list[int], x[b, prompt_len:].tolist())  # type: ignore[reportUnknownMemberType]
            stop = len(answer_toks)
            for j, tok in enumerate(answer_toks):
                if tok in stop_set:
                    stop = j
                    break
            ground_truth = extract_answer(tokenizer.decode(answer_toks[:stop]))
            if ground_truth is None:
                continue
            examples.append((x[b, :prompt_len], ground_truth))

    ctx = (
        torch.autocast(device_type=device_type, dtype=torch.bfloat16)
        if device_type == "cuda"
        else contextlib.nullcontext()
    )
    correct = 0
    for chunk_start in range(0, len(examples), gen_batch_size):
        chunk = examples[chunk_start : chunk_start + gen_batch_size]
        max_len = max(p.shape[0] for p, _ in chunk)
        input_ids = torch.full((len(chunk), max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(chunk), max_len), dtype=torch.long, device=device)
        for i, (prompt, _) in enumerate(chunk):
            L = prompt.shape[0]
            input_ids[i, max_len - L :] = prompt.to(device)
            attention_mask[i, max_len - L :] = 1
        with ctx:
            generated = model.generate(
                idx=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_k=1,
                stop_token_ids=stop_token_ids,
                attention_mask=attention_mask,
            )
        for i, (_, ground_truth) in enumerate(chunk):
            seq = cast(list[int], generated[i, max_len:].tolist())  # type: ignore[reportUnknownMemberType]
            stop = len(seq)
            for j, tok in enumerate(seq):
                if tok in stop_set:
                    stop = j
                    break
            if extract_answer(tokenizer.decode(seq[:stop])) == ground_truth:
                correct += 1
        logger.info("GSM8K progress: %d/%d", chunk_start + len(chunk), len(examples))

    model.train()
    total = len(examples)
    accuracy = correct / total if total > 0 else 0.0
    logger.info("GSM8K accuracy: %d/%d = %.4f", correct, total, accuracy)
    return accuracy
