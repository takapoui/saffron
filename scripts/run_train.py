import argparse
import json
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import torch
from torch.distributed import init_process_group

from saffron.config import DataConfig, ModelConfig, RunConfig, TrainConfig
from saffron.dataloader import DataLoader
from saffron.helpers import get_default_device
from saffron.model import Model
from saffron.optim import configure_adamw
from saffron.train import Trainer


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
    data_config: DataConfig,
) -> None:
    model = Model(model_config)
    run_config = make_run_config()
    train_loader = DataLoader(
        data_config=data_config,
        run_config=run_config,
        split="train",
    )
    val_loader = DataLoader(
        data_config=data_config,
        run_config=run_config,
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
    log_fmt = "%(asctime)s %(levelname)s %(message)s"
    Path("logs").mkdir(exist_ok=True)
    log_file = f"logs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format=log_fmt,
        handlers=[
            logging.StreamHandler(),
            RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5),
        ],
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    model_config = ModelConfig.from_dict(config["model"])
    train_config = TrainConfig.from_dict(config["train"])
    data_config = DataConfig.from_dict(config["data"])

    main(model_config=model_config, train_config=train_config, data_config=data_config)
