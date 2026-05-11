from __future__ import annotations

import contextlib
import logging
import re
from typing import TYPE_CHECKING, cast

import torch

from ..constants import LABEL_IGNORE_INDEX
from ..models import BaseModel
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
    model: BaseModel,
    tokenizer: HFTokenizer,
    val_loader: SFTDataLoader,
    device: str,
    max_new_tokens: int = 500,
) -> float:
    model.eval()

    stop_token_ids = [tokenizer.eot_token]
    if (im_end := tokenizer.im_end_token) is not None:
        stop_token_ids.append(im_end)

    correct = 0
    total = 0

    val_loader.reset()
    for _ in range(val_loader.n_steps):
        x, y = val_loader.next_batch()

        for b in range(x.shape[0]):
            y_b = y[b]
            # Find where answer starts: first position in y where label != LABEL_IGNORE_INDEX
            answer_mask = y_b != LABEL_IGNORE_INDEX
            if not answer_mask.any():
                continue
            k = int(answer_mask.nonzero(as_tuple=False)[0].item())
            # Prompt tokens are x[b, :k+1] (x is tokens[:-1], so x[:k+1] = tokens[:prompt_len])
            prompt_len = k + 1
            prompt = x[b : b + 1, :prompt_len].to(device)

            # Ground truth from stored answer tokens x[b, prompt_len:]
            answer_toks = cast(list[int], x[b, prompt_len:].tolist())  # type: ignore[reportUnknownMemberType]
            for stop_id in stop_token_ids:
                with contextlib.suppress(ValueError):
                    answer_toks = answer_toks[: answer_toks.index(stop_id)]
            ground_truth = extract_answer(tokenizer.decode(answer_toks))
            if ground_truth is None:
                continue

            # Generate completion
            generated = model.generate(
                idx=prompt,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_k=1,  # greedy
                stop_token_ids=stop_token_ids,
            )
            seq = cast(list[int], generated[0, prompt_len:].tolist())  # type: ignore[reportUnknownMemberType]
            predicted = extract_answer(tokenizer.decode(seq))

            if predicted is not None and predicted == ground_truth:
                correct += 1
            total += 1

    model.train()
    accuracy = correct / total if total > 0 else 0.0
    logger.info("GSM8K accuracy: %d/%d = %.4f", correct, total, accuracy)
    return accuracy
