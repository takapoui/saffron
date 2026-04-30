import logging

import torch


def get_default_device() -> str:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    logging.getLogger(__name__).info("Using device: %s", device)
    return device
