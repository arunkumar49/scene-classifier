from dataclasses import dataclass
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class TrainingConfig:
    """Training settings from a conf/training/ YAML file."""

    learning_rate: float
    weight_decay: float
    freeze_backbone: bool
    max_epochs: int
    patience: int  # validation checks without improvement before early stopping
    seed: int  # the training seed; data.seed only fixes the dataset split


def to_training_config(config: DictConfig) -> TrainingConfig:
    """Build a TrainingConfig from the composed Hydra config's training group."""
    values = cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))
    
    return TrainingConfig(**values)