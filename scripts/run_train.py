import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
from torch.distributed import init_process_group

from saffron.dataloader import DataLoader
from saffron.helpers import get_default_device
from saffron.model import Model, ModelConfig
from saffron.optim import configure_adamw
from saffron.train import RunConfig, TrainConfig, Trainer


def make_run_config() -> RunConfig:
    use_ddp = int(os.environ.get("RANK", -1)) != -1
    if use_ddp:
        assert torch.cuda.is_available(), "DDP training requires CUDA"
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        device_type = "cuda"
        torch.cuda.set_device(device)
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        device = get_default_device()
        device_type = torch.device(device).type

    return RunConfig(
        device=device,
        device_type=device_type,
        use_ddp=use_ddp,
        ddp_rank=ddp_rank,
        ddp_local_rank=ddp_local_rank,
        ddp_world_size=ddp_world_size,
    )


def main(
    model_config: ModelConfig,
    train_config: TrainConfig,
    data_config: dict[str, Any],
) -> None:
    model = Model(model_config)
    run_config = make_run_config()
    train_loader = DataLoader(
        B=data_config["batch_size"],
        T=data_config["seq_len"],
        data_root=Path(data_config["data_root"]),
        rank=run_config.ddp_rank,
        world_size=run_config.ddp_world_size,
        split="train",
    )
    val_loader = DataLoader(
        B=data_config["batch_size"],
        T=data_config["seq_len"],
        data_root=Path(data_config["data_root"]),
        rank=run_config.ddp_rank,
        world_size=run_config.ddp_world_size,
        split="val",
    )
    optimizer = configure_adamw(
        model=model,
        weight_decay=train_config.weight_decay,
        learning_rate=train_config.max_lr,
        device_type=run_config.device_type,
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        train_config=train_config,
        run_config=run_config,
    )
    trainer.train()

    if run_config.use_ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    model_config = ModelConfig(**config["model"])
    config["train"]["checkpoint_dir"] = Path(config["train"]["checkpoint_dir"])
    if config["train"]["resume_from"] is not None:
        config["train"]["resume_from"] = Path(config["train"]["resume_from"])
    train_config = TrainConfig(**config["train"])
    main(model_config=model_config, train_config=train_config, data_config=config["data"])
