from dataclasses import dataclass
from typing import Any, Literal, cast

from omegaconf import DictConfig, OmegaConf

# Lightning precision values our configs use; a Literal so the Trainer call type-checks
Precision = Literal["32-true", "16-mixed", "bf16-mixed"]


@dataclass(frozen=True)
class TrainerConfig:
    """Lightning Trainer hardware settings from a conf/trainer/ YAML file."""

    accelerator: str
    devices: int
    precision: Precision  # "32-true" on CPU, "16-mixed" on GPU
    limit_train_batches: int | float  # 1.0 is the whole split; an int is a batch count
    limit_val_batches: int | float
    limit_test_batches: int | float


def to_trainer_config(config: DictConfig) -> TrainerConfig:
    """Build a TrainerConfig from the composed Hydra config's trainer group."""
    values = cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))
    return TrainerConfig(**values)