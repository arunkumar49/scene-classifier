from datasets import load_dataset, DatasetDict, Dataset
import yaml
from pathlib import Path
import argparse
from typing import Any

DATA_CONFIG_PATH: str = "configs/data/hf_intel-image-classification.yaml"

class LoadData():
    
    def __init__(self, config_path: str | Path) -> None:
        self.config = self.load_config(config_path)
        self.data_dictionary = self.load_data()
        self.train_dataset: Dataset
        self.val_dataset: Dataset
        self.test_dataset: Dataset

    def get_data_config(self, key: str) -> str:
        
        return self.config[key]

    def load_config(self, config_path: str | Path) -> dict[str, Any]:
        with Path(config_path).open(encoding="utf-8") as f:
            config = yaml.safe_load(f)  # a plain dict
        return config
        
    def load_data(self) -> DatasetDict:
        dataset = load_dataset(self.config["dataset_name"])
        return dataset
    
    def split_train_test_val(self) -> None:
        split_dataset = self.data_dictionary["train"].train_test_split(test_size=0.2, seed=42)
        self.val_dataset = split_dataset.pop("test")
        self.train_dataset = split_dataset.pop("train")
        self.test_dataset = self.data_dictionary["test"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(DATA_CONFIG_PATH))
    args = parser.parse_args()

    dataset = LoadData(args.config).load_data()
    print(dataset)


if __name__ == "__main__":
    main()