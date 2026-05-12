import json
import os
from collections.abc import Iterable
from typing import cast

from datasets import load_dataset  # type: ignore[reportUnknownVariableType]

from ..tokenizer import Tokenizer
from .config import BasePrepConfig


def init_prep(
    prep_config: BasePrepConfig,
    data_files: str | None = None,
) -> tuple[Tokenizer, Iterable[dict[str, str]], str]:
    enc = Tokenizer.from_name(prep_config.tokenizer)
    fw = cast(
        Iterable[dict[str, str]],
        load_dataset(
            prep_config.dataset,
            prep_config.name if prep_config.name else None,
            data_files=data_files,
            split=prep_config.dataset_split,
        ),
    )
    os.makedirs(prep_config.data_root, exist_ok=True)
    dtype = "uint16" if enc.vocab_size <= 2**16 else "uint32"
    with open(prep_config.data_root / "meta.json", "w") as f:
        json.dump({"tokenizer": prep_config.tokenizer}, f)
    return enc, fw, dtype
