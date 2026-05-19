import argparse
import json

import torch

from saffron.config import TrainConfig
from saffron.data import DataConfig, LoaderType, PretrainDataLoader, SFTDataLoader
from saffron.helpers import make_run_config, setup_file_logging
from saffron.model import MODEL_REGISTRY, BaseConfig, BaseModel
from saffron.optim import configure_adamw
from saffron.train import SupervisedTrainer

torch.set_float32_matmul_precision("high")


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
    trainer = SupervisedTrainer(
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
