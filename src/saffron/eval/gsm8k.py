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
    evaluate_on: int = 5,  # TODO: remove cap and evaluate on full val set
) -> float:
    model.eval()

    stop_token_ids = [tokenizer.eot_token]
    if (im_end := tokenizer.im_end_token) is not None:
        stop_token_ids.append(im_end)
    pad_id = stop_token_ids[0]

    # Collect up to evaluate_on (prompt, ground_truth) pairs
    examples: list[tuple[torch.Tensor, float]] = []
    val_loader.reset()
    for _ in range(val_loader.n_steps):
        if len(examples) >= evaluate_on:
            break
        x, y = val_loader.next_batch()
        for b in range(x.shape[0]):
            if len(examples) >= evaluate_on:
                break
            y_b = y[b]
            answer_mask = y_b != LABEL_IGNORE_INDEX
            if not answer_mask.any():
                continue
            k = int(answer_mask.nonzero(as_tuple=False)[0].item())
            prompt_len = k + 1

            # Ground truth from stored answer tokens
            answer_toks = cast(list[int], x[b, prompt_len:].tolist())  # type: ignore[reportUnknownMemberType]
            for stop_id in stop_token_ids:
                with contextlib.suppress(ValueError):
                    answer_toks = answer_toks[: answer_toks.index(stop_id)]
            ground_truth = extract_answer(tokenizer.decode(answer_toks))
            if ground_truth is None:
                continue

            examples.append((x[b, :prompt_len], ground_truth))

    if not examples:
        model.train()
        return 0.0

    # Left-pad prompts into a batch and build attention mask
    max_len = max(p.shape[0] for p, _ in examples)
    input_ids = torch.full((len(examples), max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long, device=device)
    for i, (prompt, _) in enumerate(examples):
        L = prompt.shape[0]
        input_ids[i, max_len - L :] = prompt.to(device)
        attention_mask[i, max_len - L :] = 1

    # Generate entire batch in one call with autocast
    device_type = device.split(":")[0]
    ctx = (
        torch.autocast(device_type=device_type, dtype=torch.bfloat16)
        if device_type == "cuda"
        else contextlib.nullcontext()
    )
    with ctx:
        generated = model.hf_model.generate(  # type: ignore[reportUnknownMemberType]
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=stop_token_ids,
            pad_token_id=pad_id,
        )

    # Score each example — new tokens start at max_len
    correct = 0
    for i, (_, ground_truth) in enumerate(examples):
        seq = cast(list[int], generated[i, max_len:].tolist())  # type: ignore[reportUnknownMemberType]
        predicted = extract_answer(tokenizer.decode(seq))
        hit = predicted is not None and predicted == ground_truth

        # TODO: remove per-example logging
        logger.info(
            "GSM8K [%d/%d] gt=%.4f pred=%s %s",
            i + 1,
            len(examples),
            ground_truth,
            f"{predicted:.4f}" if predicted is not None else "None",
            "good" if hit else "bad",
        )

        if hit:
            correct += 1

    model.train()
    total = len(examples)
    accuracy = correct / total
    logger.info("GSM8K accuracy: %d/%d = %.4f", correct, total, accuracy)
    return accuracy
