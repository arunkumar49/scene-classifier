from dataclasses import dataclass
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class ModelConfig:
    """Model settings from a conf/model/ YAML file."""

    checkpoint: str  # Hugging Face Hub ID, e.g. "microsoft/resnet-50"


def to_model_config(config: DictConfig) -> ModelConfig:
    """Build a ModelConfig from the composed Hydra config's model group."""
    values = cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))
    return ModelConfig(**values)
