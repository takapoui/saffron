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


# Borrowed from nanochat, focused on lambdalabs gpus
_PEAK_FLOPS_TABLE: tuple[tuple[list[str], float], ...] = (
    # NVIDIA Blackwell
    (["b200"], 2.25e15),
    # NVIDIA Hopper — GH200 reports as H100 SXM5
    (["h100", "pcie"], 756e12),
    (["h100"], 989e12),  # SXM5
    # NVIDIA Ampere data center
    (["a100"], 312e12),  # A100 PCIe / SXM4
    (["a6000"], 77.4e12),  # RTX A6000 48GB (Ampere)
    (["a10"], 125e12),  # A10 24GB PCIe
    # NVIDIA Ada data center
    (["6000"], 91.1e12),  # RTX 6000 Ada 24GB
    # NVIDIA Volta — no native BF16, FP16 peak listed
    (["v100"], 125e12),
)


def get_peak_flops(device_type: str) -> float:
    if device_type != "cuda":
        return float("inf")
    device_name = torch.cuda.get_device_name(0).lower()
    for keywords, flops in _PEAK_FLOPS_TABLE:
        if all(kw in device_name for kw in keywords):
            return flops
    return float("inf")
