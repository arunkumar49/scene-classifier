
from typing import Callable
from pathlib import Path
import lightning as L
from torch.utils.data import DataLoader
from scene_classifier.data.scene_dataset import SceneDataset
from scene_classifier.utils.data_config import DataConfig
from scene_classifier.utils.s3_util import download_split


class DataModule(L.LightningDataModule):
    
    def __init__(self, config: DataConfig, train_transform: Callable, eval_transform: Callable):
        super().__init__()
        self.config = config
        self.train_transform = train_transform  # with augmentation
        self.eval_transform = eval_transform  # for val and test, no augmentation
        self.local_dir = Path(config.local_dir)
    
    
    def prepare_data(self) -> None:
        # Runs once per node: download only, don't assign state here.
        for split in (self.config.train_split, self.config.test_split, self.config.val_split):
            download_split(
                self.config.s3.bucket, 
                self.config.s3.prefix, 
                split,
                self.local_dir,
                self.config.image_suffixes,
                max_workers=self.config.s3.max_workers,
                region=self.config.s3.region,
                profile=self.config.s3.profile,
            )
    
    def setup(self, stage: str) -> None:
        # Runs on every process: build the datasets from the downloaded files
        if stage == "fit":
            self.train_dataset = self._build_dataset(self.config.train_split, self.train_transform)
        if stage in ("fit", "validate"):
            self.val_dataset = self._build_dataset(self.config.val_split, self.eval_transform)
        if stage == "test":
            self.test_dataset = self._build_dataset(self.config.test_split, self.eval_transform)


    def train_dataloader(self) -> DataLoader:
        return self._build_dataloader(self.train_dataset, shuffle=True)


    def val_dataloader(self) -> DataLoader:
        return self._build_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._build_dataloader(self.test_dataset, shuffle=False)

    def teardown(self, stage: str) -> None:
        # Nothing to clean up: downloaded files are kept, so the next run skips them
        ...

    def _build_dataset(self, split: str, transform: Callable) -> SceneDataset:
        root = self.local_dir / split
        dataset = SceneDataset(root, self.config.image_suffixes, transform)
        if len(dataset) == 0:
            raise ValueError(f"No images found in {root}.")

        # Folder names are class IDs, so each one must be an index into config.classes
        labels = {label for _, label in dataset.samples}
        unknown = labels - set(range(len(self.config.classes)))
        if unknown:
            raise ValueError(
                f"{root} has class IDs {sorted(unknown)}, "
                f"but the config has only {len(self.config.classes)} classes."
            )
        return dataset

    def _build_dataloader(self, dataset: SceneDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            # Keep workers alive between epochs; starting them is slow on Windows
            persistent_workers=self.config.num_workers > 0,
        )
    
    
    # def show_sample(self, split: Dataset, idx: int) -> tuple[str, JpegImageFile]:
    #     """ 
    #     Return a image and its label from given dataset and index.
    #     This funtion is only used by training purpose.
    #     """
    #     sample = split[idx]
    #     label = sample["label"]
    #     label_feature = split.features["label"]
    #     # ClassLabel features can map the integer id back to its class name
    #     if isinstance(label_feature, ClassLabel):
    #         label = f"{label} ({label_feature.int2str(label)})"
        
    #     return (f"index={idx}  label={label}  size={sample['image'].size}", sample["image"])