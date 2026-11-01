from dataclasses import dataclass
from typing import Any, cast
from omegaconf import DictConfig, OmegaConf



@dataclass(frozen=True)
class TrackingConfig():
    experiment_name: str
    run_name: str
    tracking_uri: str
    log_model: bool
    git_sha: str | None
    

def to_tracking_config(config: DictConfig) -> TrackingConfig:
    result = cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))
    
    return TrackingConfig(**result)