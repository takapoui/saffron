import logging

import numpy as np
import torch


def get_default_device() -> str:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    logging.getLogger(__name__).info("Using device: %s", device)
    return device


def get_peak_flops(device_type: str) -> float:
    if device_type != "cuda":
        # meaningless
        return np.inf
    device_name = torch.cuda.get_device_name(0).lower()
    if "h100" in device_name:
        return 989e12
    elif "a100" in device_name:
        return 312e12
    elif "a10" in device_name:
        return 31.2e12
    elif "6000" in device_name:
        return 91.1e12
    else:
        return np.inf
