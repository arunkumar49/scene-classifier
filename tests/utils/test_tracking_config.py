from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf.errors import InterpolationResolutionError

from scene_classifier.utils.tracking_config import TrackingConfig, to_tracking_config

CONF_DIR = str(Path(__file__).resolve().parents[2] / "conf")


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("GIT_COMMIT_SHA", raising=False)


def load(*overrides: str) -> TrackingConfig:
    with initialize_config_dir(version_base="1.3", config_dir=CONF_DIR):
        config = compose("config", overrides=list(overrides), return_hydra_config=True)
        # @hydra.main does this; without it ${hydra:runtime.choices...} can't resolve
        HydraConfig.instance().set_config(config)
        return to_tracking_config(config.tracking)


def test_local_falls_back_to_sqlite() -> None:
    tracking = load("tracking=local")
    assert tracking.tracking_uri == "sqlite:///mlflow.db"
    assert tracking.git_sha is None
    assert tracking.log_model is False


def test_local_uses_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "arn:example")
    monkeypatch.setenv("GIT_COMMIT_SHA", "abc123")
    tracking = load("tracking=local")
    assert tracking.tracking_uri == "arn:example"
    assert tracking.git_sha == "abc123"


@pytest.mark.parametrize(
    ("overrides", "experiment_name", "run_name"),
    [
        ((), "scene-classifier-resnet50", "head_only"),
        (("model=vit_base", "training=full_finetune"), "scene-classifier-vit_base", "full_finetune"),
    ],
)
def test_names_follow_config_choices(
    overrides: tuple[str, ...], experiment_name: str, run_name: str
) -> None:
    tracking = load(*overrides)
    assert tracking.experiment_name == experiment_name
    assert tracking.run_name == run_name


def test_sagemaker_requires_tracking_uri() -> None:
    with pytest.raises(InterpolationResolutionError):
        load("tracking=sagemaker")
