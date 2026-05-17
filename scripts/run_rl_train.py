"""Entrypoint for GRPO training. Mirrors scripts/run_train.py for SFT/pretraining."""

import argparse
import json
from pathlib import Path

import torch

from saffron.config import RLConfig
from saffron.data import RLDataLoader
from saffron.helpers import make_run_config, setup_file_logging
from saffron.model import MODEL_REGISTRY, BaseConfig, BaseModel
from saffron.optim import configure_adamw
from saffron.rl.trainer import RLTrainer

torch.set_float32_matmul_precision("high")


def main(
    model_config: BaseConfig,
    rl_config: RLConfig,
    model_cls: type[BaseModel],
    data_root: Path,
) -> None:
    model = model_cls(model_config)
    ref_model = model_cls(model_config)
    run_config = make_run_config()

    train_loader = RLDataLoader(
        path=data_root / "train.jsonl",
        tokenizer=model.tokenizer,
    )
    val_loader = RLDataLoader(
        path=data_root / "val.jsonl",
        tokenizer=model.tokenizer,
    )
    optimizer = configure_adamw(
        model=model,
        config=rl_config.optimizer,
        device_type=run_config.device_type,
    )
    trainer = RLTrainer(
        model=model,
        ref_model=ref_model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        rl_config=rl_config,
        run_config=run_config,
    )
    trainer.train()

    if run_config.use_ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    setup_file_logging("rl_train")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    model_type = config["model"]["model_type"]
    model_cls = MODEL_REGISTRY[model_type]
    model_config = model_cls.config_class.from_dict(config["model"])
    rl_config = RLConfig.from_dict(config["train"])
    data_root = Path(config["data"]["data_root"])

    main(
        model_config=model_config,
        rl_config=rl_config,
        model_cls=model_cls,
        data_root=data_root,
    )
