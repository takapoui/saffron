from .base_model import BaseModel
from .config import BaseConfig, GPT2Config, HFConfig, NeoGPTConfig
from .gpt2 import GPT2
from .hf import HFModel
from .neogpt import NeoGPT

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "gpt2": GPT2,
    "hf": HFModel,
    "neogpt": NeoGPT,
}

__all__ = [
    "MODEL_REGISTRY",
    "BaseConfig",
    "BaseModel",
    "GPT2",
    "GPT2Config",
    "HFConfig",
    "HFModel",
    "NeoGPT",
    "NeoGPTConfig",
]
