from functools import partial
from typing import Any

import torch
from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms.v2 import InterpolationMode
from transformers import AutoImageProcessor

from scene_classifier.utils.model_config import ModelConfig


# Train-only augmentation, hardcoded until the Hydra config exists
CROP_SCALE = (0.35, 1.0)        # scenes tolerate fairly tight crops
HFLIP_P = 0.5
RANDAUG_OPS, RANDAUG_MAG = 2, 7  # moderate; lower the magnitude for short fine-tunes
ERASE_P = 0.25



def load_processor(model_cfg: ModelConfig) -> Any:
    # use_fast=True so processor.size is a SizeDict with attribute access
    return AutoImageProcessor.from_pretrained(model_cfg.checkpoint, use_fast=True)


def _interpolation(processor: Any) -> InterpolationMode:
    return InterpolationMode[Image.Resampling(processor.resample).name]


def _resize_steps(processor: Any, checkpoint: str) -> list[v2.Transform]:
    """
    Return the resize (and centre-crop) steps that match the checkpoint's own processor.
    """
    # Any: each concrete processor sets size and resample; no shared base declares them
    # resample is an int for some processors and a PIL Resampling enum for others
    resize = partial(v2.Resize, interpolation=_interpolation(processor), antialias=True)

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


def _output_size(processor: Any) -> tuple[int, int]:
    """Final (H, W) the model sees, so the train crop matches eval."""
    size = processor.size
    if size.height is not None and size.width is not None:
        return size.height, size.width
    crop = getattr(processor, "crop_size", None)
    if crop is not None and getattr(processor, "do_center_crop", False):
        return crop.height, crop.width
    return size.shortest_edge, size.shortest_edge


def build_eval_transforms(model_cfg: ModelConfig) -> v2.Compose:
    """
    Build the validation, test and serving transforms: the checkpoint's own
    preprocessing, without augmentation.
    """
    processor = load_processor(model_cfg)
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
    """Training: random crop at the eval output size, flip, RandAugment, random erasing."""
    processor = load_processor(model_cfg)
    
    interp = _interpolation(processor)
    
    # Fill uncovered pixels with the dataset mean (uint8) instead of black
    mean_fill = [round(m * 255) for m in processor.image_mean]
    
    return v2.Compose(
        [
            v2.RGB(),
            v2.ToImage(),
            v2.RandomResizedCrop(
                _output_size(processor), scale=CROP_SCALE, interpolation=interp, antialias=True
            ),
            # *_resize_steps(processor, model_cfg.checkpoint),
            # v2.RandomHorizontalFlip(p=HFLIP_P),
            # v2.RandomRotation((-ROTATION_DEGREES, ROTATION_DEGREES)),
            # v2.ColorJitter(**COLOR_JITTER),
            v2.RandAugment(
                num_ops=RANDAUG_OPS, magnitude=RANDAUG_MAG, interpolation=interp, fill=mean_fill
            ),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=processor.image_mean, std=processor.image_std),
            v2.RandomErasing(p=ERASE_P, scale=(0.02, 0.2)),  # tensor op, so after Normalize
        ]
    )
