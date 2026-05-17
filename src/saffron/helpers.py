import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import torch
import wandb


def setup_file_logging(prefix: str) -> None:
    """Log to both stdout and logs/{prefix}_{timestamp}.log."""
    Path("logs").mkdir(exist_ok=True)
    log_file = f"logs/{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5),
        ],
    )


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


def init_wandb(project: str, config_dict: dict[str, Any]) -> None:
    """Initialize wandb with a project and config dict. Caller must gate on master_process."""
    wandb.init(project=project, config=config_dict)


def format_metric_line(step: int, metrics: dict[str, float]) -> str:
    """One-line formatted string for stdout/file logging."""

    def _fmt(key: str, val: float) -> str:
        return f"{key}: {val:.2e}" if key == "lr" else f"{key}: {val:.4f}"

    parts = [f"step: {step:5d}"] + [_fmt(k, v) for k, v in metrics.items()]
    return " | ".join(parts)
