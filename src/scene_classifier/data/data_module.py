from datasets import ClassLabel, Dataset
from lightning import LightningDataModule
from PIL.JpegImagePlugin import JpegImageFile


class DataModule(LightningDataModule):
    
    
    def show_sample(self, split: Dataset, idx: int) -> tuple[str, JpegImageFile]:
        """ 
        Return a image and its label from given dataset and index.
        This funtion is only used by training purpose.
        """
        sample = split[idx]
        label = sample["label"]
        label_feature = split.features["label"]
        # ClassLabel features can map the integer id back to its class name
        if isinstance(label_feature, ClassLabel):
            label = f"{label} ({label_feature.int2str(label)})"
        
        return (f"index={idx}  label={label}  size={sample['image'].size}", sample["image"])