import argparse
import json
import os

import torch
from torch.distributed import init_process_group

from saffron.config import RunConfig, TrainConfig
from saffron.data import DataConfig, LoaderType, PretrainDataLoader, SFTDataLoader
from saffron.helpers import get_default_device, setup_file_logging
from saffron.model import GPT2, BaseConfig, BaseModel, HFModel
from saffron.optim import configure_adamw
from saffron.train import Trainer

torch.set_float32_matmul_precision("high")

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "gpt2": GPT2,
    "hf": HFModel,
}


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
    model_config: BaseConfig,
    train_config: TrainConfig,
    data_config: DataConfig,
    model_cls: type[BaseModel],
) -> None:
    model = model_cls(model_config)
    run_config = make_run_config()
    loader_cls = SFTDataLoader if data_config.loader_type == LoaderType.SFT else PretrainDataLoader
    train_loader = loader_cls(
        data_config=data_config,
        run_config=run_config,
        split="train",
    )
    val_loader = loader_cls(
        data_config=data_config,
        run_config=run_config,
        split="val",
    )
    optimizer = configure_adamw(
        model=model,
        config=train_config.optimizer,
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
    setup_file_logging("train")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    model_type = config["model"]["model_type"]
    model_cls = MODEL_REGISTRY[model_type]
    model_config = model_cls.config_class.from_dict(config["model"])
    train_config = TrainConfig.from_dict(config["train"])
    data_config = DataConfig.from_dict(config["data"])

    main(
        model_config=model_config,
        train_config=train_config,
        data_config=data_config,
        model_cls=model_cls,
    )
