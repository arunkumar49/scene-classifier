import hydra
from omegaconf import DictConfig

from scene_classifier.data.data_cli import LoadData
from scene_classifier.utils.data_config import to_data_config


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Load the Hub dataset, check its classes and print the train/val/test splits."""
    data = LoadData(to_data_config(cfg.data))  # __init__ runs load_data()
    data.split_train_test_val()


if __name__ == "__main__":
    main()
