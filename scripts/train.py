import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from scene_classifier.training.train import train


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Fine-tune the configured model and test its best checkpoint."""
    output_dir = HydraConfig.get().runtime.output_dir  # Hydra's per-run folder
    train(cfg, output_dir)


# Windows DataLoader workers re-import this script; the guard stops them from training
if __name__ == "__main__":
    main()
