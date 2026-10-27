from typing import Any, Callable
from pathlib import Path
import torch
from PIL import Image
from scene_classifier.utils.data_config import DataConfig
from torch.utils.data import Dataset



class SceneDataset(Dataset):
    """
    Reads {root}/{class_id}/{image}.jpg, where class_id is an integer folder name.
    """
    
    def __init__(self, root: str | Path, image_suffixes: list[str], transform: Callable | None = None):
        self.transform = transform
        self.samples = []
        
        for class_dir in sorted(Path(root).iterdir()):
            if not class_dir.is_dir():
                continue
            
            label = int(class_dir.name)
            
            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() in image_suffixes:
                    self.samples.append((image_path, label))
    
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, index: int) -> tuple[Image.Image | torch.Tensor, int]:
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label