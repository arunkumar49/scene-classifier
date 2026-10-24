from dataclasses import dataclass
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class S3UploadConfig:
    """Where and how to upload one dataset version."""

    bucket: str
    prefix: str  # the base prefix, e.g. "scene-classifier/data/processed"
    dataset_version: str  # e.g. "v1"
    region: str | None = None
    kms_key_id: str | None = None  # None relies on the bucket's default encryption
    max_workers: int = 32
    profile: str | None = None  # None uses the default AWS credentials
    overwrite: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class DataConfig:
    """Dataset settings from a conf/data/ YAML file."""

    dataset_name: str
    revision: str
    train_split: str
    val_split: str
    test_split: str
    classes: list[str]  # class names in class ID order (PRD 7.2)
    val_size: float
    seed: int
    s3: S3UploadConfig


def to_data_config(config: DictConfig) -> DataConfig:
    """Build a DataConfig from a data config, e.g. the composed Hydra config's data group."""
    # Plain dicts and lists, so the dataclasses never hold OmegaConf types
    values = cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))
    values.pop("_target_", None)  # Hydra metadata, not a setting
    s3 = S3UploadConfig(**values.pop("s3"))
    return DataConfig(**values, s3=s3)





