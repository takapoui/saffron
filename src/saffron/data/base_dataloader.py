import json
import logging
from abc import ABC, abstractmethod

import torch

from ..config import RunConfig
from .config import DataConfig

logger = logging.getLogger(__name__)


class BaseDataLoader(ABC):
    def __init__(
        self,
        data_config: DataConfig,
        run_config: RunConfig,
        split: str,
    ) -> None:
        # The files are already tokenized
        self.B = data_config.batch_size
        self.T = data_config.seq_len
        self.rank = run_config.ddp_rank

        meta_path = data_config.data_root / "meta.json"
        assert meta_path.exists(), (
            f"No meta.json found in {data_config.data_root}. Re-run data prep to generate it."
        )
        with open(meta_path) as f:
            meta = json.load(f)
        self.tokenizer_name: str = meta["tokenizer"]
        if self.tokenizer_name != data_config.tokenizer:
            raise ValueError(
                f"Data in '{data_config.data_root}' was built with tokenizer "
                f"'{self.tokenizer_name}' but config specifies '{data_config.tokenizer}'. "
                "Re-run data prep with the correct tokenizer."
            )
        self.world_size = run_config.ddp_world_size

        self.shards = sorted(
            [str(p) for p in (data_config.data_root / split).iterdir() if p.name.endswith(".npy")]
        )
        assert len(self.shards) > 0, f"No shards found in {data_config.data_root} for split {split}"
        if self.rank == 0:
            logger.info(
                f"Found {len(self.shards)} shards for split {split} in {data_config.data_root}"
            )

        self.reset()

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def advance(self, tokens: int) -> None:
        pass

    @abstractmethod
    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        pass
