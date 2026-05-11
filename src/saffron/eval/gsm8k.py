from __future__ import annotations

import contextlib
import logging
import re
from typing import TYPE_CHECKING, cast

import torch

from ..constants import LABEL_IGNORE_INDEX
from ..models.hf import HFModel
from ..tokenizer import HFTokenizer

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
    model: HFModel,
    tokenizer: HFTokenizer,
    val_loader: SFTDataLoader,
    device: str,
    max_new_tokens: int = 500,
    gen_batch_size: int = 64,
) -> float:
    model.eval()

    stop_token_ids = [tokenizer.eot_token]
    if (im_end := tokenizer.im_end_token) is not None:
        stop_token_ids.append(im_end)
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
            for stop_id in stop_token_ids:
                with contextlib.suppress(ValueError):
                    answer_toks = answer_toks[: answer_toks.index(stop_id)]
            ground_truth = extract_answer(tokenizer.decode(answer_toks))
            if ground_truth is None:
                continue
            examples.append((x[b, :prompt_len], ground_truth))

    # Generate in batches
    device_type = device.split(":")[0]
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
            generated = model.hf_model.generate(  # type: ignore[reportUnknownMemberType]
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=stop_token_ids,
                pad_token_id=pad_id,
            )

        for i, (_, ground_truth) in enumerate(chunk):
            seq = cast(list[int], generated[i, max_len:].tolist())  # type: ignore[reportUnknownMemberType]
            predicted = extract_answer(tokenizer.decode(seq))
            if predicted is not None and predicted == ground_truth:
                correct += 1

        logger.info("GSM8K progress: %d/%d", chunk_start + len(chunk), len(examples))

    model.train()
    total = len(examples)
    accuracy = correct / total if total > 0 else 0.0
    logger.info("GSM8K accuracy: %d/%d = %.4f", correct, total, accuracy)
    return accuracy
