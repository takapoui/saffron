import logging
import os

import numpy as np

from ..constants import LABEL_IGNORE_INDEX
from ..tokenizer import HFTokenizer
from ._prep_utils import init_prep
from .config import SFTPrepConfig

logger = logging.getLogger(__name__)


def format_example(
    example: dict[str, str], enc: HFTokenizer, system_prompt: str | None
) -> tuple[list[int], int]:
    if system_prompt is None:
        prompt_only = []
        messages = []
    else:
        prompt_only = [{"role": "system", "content": system_prompt}]
        messages = [{"role": "system", "content": system_prompt}]

    prompt_only.append(
        {"role": "user", "content": example["question"]},
    )

    messages.extend(
        [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ]
    )
    prompt_ids = enc.apply_chat_template(
        prompt_only,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    full_ids = enc.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
    )
    return full_ids, len(prompt_ids)


def prepare_sft_dataset(prep_config: SFTPrepConfig) -> None:
    enc, fw, dtype = init_prep(prep_config)
    assert isinstance(enc, HFTokenizer)
    split_dir = prep_config.data_root / prep_config.output_split.value
    os.makedirs(split_dir, exist_ok=True)

    shard_examples: list[tuple[list[int], list[int]]] = []  # (tokens, labels) per example
    shard_index = 0

    def _write(examples: list[tuple[list[int], list[int]]], idx: int, dtype: str) -> None:
        n = len(examples)
        tokens_arr = np.full((n, prep_config.max_length), enc.eot_token, dtype=dtype)
        labels_arr = np.full((n, prep_config.max_length), LABEL_IGNORE_INDEX, dtype=np.int32)
        for i, (tokens, labels) in enumerate(examples):
            tokens_arr[i, : len(tokens)] = tokens
            labels_arr[i, : len(tokens)] = labels
        np.save(split_dir / f"{idx:06d}_tokens", tokens_arr)
        np.save(split_dir / f"{idx:06d}_labels", labels_arr)

    for example in fw:
        tokens, prompt_len = format_example(example, enc, system_prompt=prep_config.system_prompt)

        if prompt_len >= prep_config.max_length:
            logger.info("Skip: prompt alone exceeds max_length.")
            continue

        labels = [LABEL_IGNORE_INDEX] * prompt_len + tokens[prompt_len:]

        if len(tokens) > prep_config.max_length:
            logger.info(f"Truncate example with length {len(tokens)} to {prep_config.max_length}")
            tokens = tokens[: prep_config.max_length]
            labels = labels[: prep_config.max_length]
        shard_examples.append((tokens, labels))

        if len(shard_examples) == prep_config.examples_per_shard:
            _write(shard_examples, shard_index, dtype)
            shard_index += 1
            shard_examples = []

    # Write the leftover
    if len(shard_examples) > 0:
        _write(shard_examples, shard_index, dtype)
