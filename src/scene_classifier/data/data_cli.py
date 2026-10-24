
from datasets import ClassLabel, Dataset, DatasetDict, load_dataset

from scene_classifier.utils.data_config import DataConfig


class LoadData:

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.data_dictionary = self.load_data()
        self.train_dataset: Dataset
        self.val_dataset: Dataset
        self.test_dataset: Dataset

    def load_data(self) -> DatasetDict:
        dataset = load_dataset(self.config.dataset_name, revision=self.config.revision)
        self.check_classes(dataset)
        return dataset

    def check_classes(self, dataset: DatasetDict) -> None:
        """
        Raise if any split's labels differ from the fixed classes in the config (PRD 7.2).
        """
        for split, split_dataset in dataset.items():
            label_feature = split_dataset.features["label"]
            # ClassLabel features hold the class names in class ID order
            if not isinstance(label_feature, ClassLabel):
                raise TypeError(f"The 'label' column of the '{split}' split must be a ClassLabel feature.")
            if label_feature.names != self.config.classes:
                raise ValueError(
                    f"The '{split}' split has classes {label_feature.names}, "
                    f"but the config expects {self.config.classes}."
                )

    def split_train_test_val(self) -> None:
        split_dataset = self.data_dictionary[self.config.train_split].train_test_split(
            test_size=self.config.val_size, seed=self.config.seed
        )
        self.val_dataset = split_dataset.pop("test")
        self.train_dataset = split_dataset.pop("train")
        self.test_dataset = self.data_dictionary[self.config.test_split]
