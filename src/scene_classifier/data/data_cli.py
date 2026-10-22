from datasets import load_dataset, DatasetDict, Dataset
from pathlib import Path
import argparse
from scene_classifier.utils.data_config import DATA_CONFIG_PATH, DataConfig, load_data_config, parse_data_config



class LoadData():

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.data_dictionary = self.load_data()
        self.train_dataset: Dataset
        self.val_dataset: Dataset
        self.test_dataset: Dataset

    def load_data(self) -> DatasetDict:
        dataset = load_dataset(self.config.dataset_name, revision=self.config.revision)
        return dataset

    def split_train_test_val(self) -> None:
        split_dataset = self.data_dictionary[self.config.train_split].train_test_split(
            test_size=self.config.val_size, seed=self.config.seed
        )
        self.val_dataset = split_dataset.pop("test")
        self.train_dataset = split_dataset.pop("train")
        self.test_dataset = self.data_dictionary[self.config.test_split]


def main() -> None:
    
    data = LoadData(parse_data_config())  # __init__ runs load_data()
    data.split_train_test_val()
    print(data.train_dataset)
    print(data.val_dataset)
    print(data.test_dataset)


if __name__ == "__main__":
    main()