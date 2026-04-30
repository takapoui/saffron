import inspect
import logging
import math

import torch

from .model import Model

logger = logging.getLogger(__name__)


def configure_adamw(
    model: Model,
    weight_decay: float,
    learning_rate: float,
    device: str,
) -> torch.optim.AdamW:
    param_dict = {name: param for name, param in model.named_parameters() if param.requires_grad}
    decay_params = [param for param in param_dict.values() if param.ndim >= 2]
    no_decay_params = [param for param in param_dict.values() if param.ndim <= 1]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    logger.info(
        f"decayed parameters: {len(decay_params)} groups, "
        f"{sum(param.numel() for param in decay_params)}"
    )
    logger.info(
        f"nondecayed parameters: {len(no_decay_params)} groups, "
        f"{sum(param.numel() for param in no_decay_params)}"
    )

    device_type = torch.device(device).type
    use_fused = device_type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters
    logger.info(f"using fused AdamW: {use_fused}")
    return torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=use_fused,
    )


def get_lr_cosine(step: int, max_steps: int, max_lr: float, warmup_steps: int) -> float:
    min_lr = max_lr * 0.1
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
