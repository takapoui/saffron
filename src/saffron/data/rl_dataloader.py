import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class RLDataLoader:
    # Note that this does not inherit from BaseDataLoader because the data shape is
    # different: variable-length prompts plus heterogeneous metadata, not fixed-shape tensors.

    def __init__(self, path: Path, tokenizer_name: str) -> None:
        self.tokenizer_name = tokenizer_name
        self._validate_meta(path.parent)
        with open(path) as f:
            self.examples = [json.loads(line) for line in f]
            logger.info(f"Loaded {len(self.examples)} examples into memory.")

    def _validate_meta(self, data_root: Path) -> None:
        meta_path = data_root / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No meta.json found in {data_root}. Re-run data prep to generate it."
            )
        with open(meta_path) as f:
            meta = json.load(f)
        if meta["tokenizer"] != self.tokenizer_name:
            raise ValueError(
                f"Data in '{data_root}' was built with tokenizer "
                f"'{meta['tokenizer']}' but config specifies '{self.tokenizer_name}'. "
                "Re-run data prep with the correct tokenizer."
            )

    def __len__(self) -> int:
        return len(self.examples)

    def sample_batch(self, n: int, rng: np.random.Generator) -> list[dict[str, Any]]:
        indices = rng.choice(len(self), size=n, replace=False)
        return [self.examples[i] for i in indices]
