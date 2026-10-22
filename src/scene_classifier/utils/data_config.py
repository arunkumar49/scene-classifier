import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Any
import argparse

DATA_CONFIG_PATH: str = "configs/data/hf_intel-image-classification.yaml"


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
    """Dataset settings from a configs/data/ YAML file."""

    dataset_name: str
    revision: str
    train_split: str
    val_split: str
    test_split: str
    val_size: float
    seed: int
    s3: S3UploadConfig


def load_data_config(config_path: str | Path = DATA_CONFIG_PATH) -> DataConfig:
    with Path(config_path).open(encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)  # a plain dict
    config.pop("_target_", None)  # Hydra metadata, not a setting
    s3 = S3UploadConfig(**config.pop("s3"))
    return DataConfig(**config, s3=s3)




def parse_data_config() -> DataConfig:
    """Load the DataConfig named by --config, or the default."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(DATA_CONFIG_PATH))
    args = parser.parse_args()
    return load_data_config(args.config)
