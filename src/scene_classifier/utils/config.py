from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def load_config(config_dir: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """
    Compose conf/config.yaml with Hydra's Compose API, for notebooks and tests.
    Scripts use @hydra.main instead.
    """
    # initialize_config_dir needs an absolute path
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(config_dir).resolve())):
        return compose(config_name="config", overrides=overrides or [])
