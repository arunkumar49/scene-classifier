from functools import partial
from typing import Any

import torch
from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms.v2 import InterpolationMode
from transformers import AutoImageProcessor

from scene_classifier.utils.model_config import ModelConfig

# Train-only augmentation, hardcoded until the Hydra config exists
HFLIP_P = 0.5
ROTATION_DEGREES = 10
COLOR_JITTER = {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2}


def _resize_steps(processor: Any, checkpoint: str) -> list[v2.Transform]:
    """
    Return the resize (and centre-crop) steps that match the checkpoint's own processor.
    """
    # Any: each concrete processor sets size and resample; no shared base declares them
    # resample is an int for some processors and a PIL Resampling enum for others
    interpolation = InterpolationMode[Image.Resampling(processor.resample).name]
    resize = partial(v2.Resize, interpolation=interpolation, antialias=True)

    # size is a SizeDict whose unset fields are None, so read attributes, not keys
    size = processor.size
    if size.height is not None and size.width is not None:
        # e.g. ViT: resize straight to height x width
        return [resize((size.height, size.width))]

    shortest_edge = size.shortest_edge
    crop_pct = getattr(processor, "crop_pct", None)
    # ConvNext-style processors skip the crop from 384 px up, which isn't handled here
    if shortest_edge is not None and crop_pct is not None and shortest_edge < 384:
        # e.g. ResNet: resize the shortest edge to 224 / 0.875 = 256, then crop to 224
        return [resize(int(shortest_edge / crop_pct)), v2.CenterCrop(shortest_edge)]

    raise ValueError(
        f"Unsupported image processor size {size} for checkpoint '{checkpoint}'."
    )


def build_eval_transforms(model_cfg: ModelConfig) -> v2.Compose:
    """
    Build the validation, test and serving transforms: the checkpoint's own
    preprocessing, without augmentation.
    """
    processor = AutoImageProcessor.from_pretrained(model_cfg.checkpoint)
    return v2.Compose(
        [
            v2.RGB(),  # greyscale, RGBA or palette images -> 3 channels
            v2.ToImage(),  # PIL -> tv_tensors.Image (uint8)
            *_resize_steps(processor, model_cfg.checkpoint),
            v2.ToDtype(torch.float32, scale=True),  # uint8 [0, 255] -> float [0, 1]
            v2.Normalize(mean=processor.image_mean, std=processor.image_std),
        ]
    )


def build_train_transforms(model_cfg: ModelConfig) -> v2.Compose:
    """
    Build the training transforms: the eval preprocessing plus a random horizontal
    flip, small rotation and colour jitter.
    """
    processor = AutoImageProcessor.from_pretrained(model_cfg.checkpoint)
    return v2.Compose(
        [
            v2.RGB(),
            v2.ToImage(),
            *_resize_steps(processor, model_cfg.checkpoint),
            v2.RandomHorizontalFlip(p=HFLIP_P),
            v2.RandomRotation((-ROTATION_DEGREES, ROTATION_DEGREES)),
            v2.ColorJitter(**COLOR_JITTER),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=processor.image_mean, std=processor.image_std),
        ]
    )
