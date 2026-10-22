"""Upload an image classification dataset to S3 as ``<split>/<class>/<image>`` objects.

Images are uploaded with their original encoded bytes, never decoded and re-encoded,
so the objects in S3 are identical to the source files. After every image is in S3,
a ``manifest.csv`` (one row per image) and a ``_metadata.json`` are written.
``_metadata.json`` is written last and marks the dataset version as complete.
"""


from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http import client
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
import boto3
import PIL.Image
from botocore.config import Config
from datasets import ClassLabel, Dataset, DatasetDict, Image
from moto import kms
from tqdm import tqdm
import io
import mimetypes
from scene_classifier.utils.data_config import S3UploadConfig

if TYPE_CHECKING:
    # boto3-stubs is a dev dependency, so only import it for the type checker
    from mypy_boto3_s3 import S3Client
    from mypy_boto3_s3.type_defs import PutObjectRequestTypeDef


@dataclass(frozen=True)
class ImageRecord:
    """One encoded image and its place in the split/class hierarchy."""

    split: str
    label_id: int
    label: str
    filename: str
    data: bytes
    source: str  # the original file path, or "<split>[<index>]" for bytes-only rows


@dataclass
class UploadReport:
    """Outcome of an upload. ``complete`` is True only once every image is in S3."""

    uploaded: int = 0
    skipped: int = 0
    # S3 key -> error
    failed: dict[str, str] = field(default_factory=dict)
    # split -> class -> number of images
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    complete: bool = False

    @property
    def total(self) -> int:
        return sum(sum(per_class.values()) for per_class in self.counts.values())




def upload_dataset_to_s3(
    dataset: Dataset,
    split: str,
    s3_config: S3UploadConfig,
    s3_client: "S3Client | None" = None,
) -> None:
    """
    Uploads one split of an image dataset to S3, one image at a time, maintaining:
    bucket / split / class_name / image_name.jpg

    Images are uploaded with their original file bytes, never decoded and re-encoded.

    Arguments:
    - dataset: Hugging Face Dataset with an 'image' column and a ClassLabel 'label' column
    - split: split name used in the S3 key, e.g. 'train', 'val' or 'test'
    - s3_config: bucket and AWS profile to upload with
    - s3_client: boto3 S3 client; defaults to a client for s3_config.profile.
      When given, s3_config.profile is ignored.
    """
    if s3_client is None:
        s3_client = get_s3_client(s3_config.profile)

    # label_feature = dataset.features["label"]
    # # ClassLabel features can map the integer id back to its class name
    # if not isinstance(label_feature, ClassLabel):
    #     raise TypeError("The 'label' column must be a ClassLabel feature.")

    # decode=False gives each image as {'bytes': ..., 'path': ...} instead of a PIL Image
    raw_dataset = dataset.cast_column('image', Image(decode=False))

    # Dataset.__iter__ is untyped; each item is a dict of column -> value
    for item in cast(Iterable[dict[str, Any]], raw_dataset):
        # Extract fields from your dataset item
        image = item['image']
        path = image['path']
        if not path:
            raise ValueError("Every image needs a file path to name its S3 object.")
        # class_name = label_feature.names[item['label']]  # e.g., 'forest'
        image_name = Path(path).name                     # e.g., '0.jpg'

        # Original file bytes: from the dataset if it holds them, otherwise from disk
        body = image['bytes'] if image['bytes'] is not None else Path(path).read_bytes()

        # Construct hierarchical S3 key (path)
        s3_key = f"data/raw/{split}/{item['label']}/{image_name}"

        # Upload to S3
        request: "PutObjectRequestTypeDef" = {
            'Bucket': s3_config.bucket,
            'Key': s3_key,
            'Body': body,
            'ContentType': mimetypes.guess_type(image_name)[0] or 'application/octet-stream',
        }
        # Without a key, send no SSE headers so the bucket's default encryption applies
        if s3_config.kms_key_id:
            request['ServerSideEncryption'] = 'aws:kms'
            request['SSEKMSKeyId'] = s3_config.kms_key_id
        s3_client.put_object(**request)




def get_s3_client(profile: str | None = None) -> "S3Client":
    return boto3.Session(profile_name=profile).client("s3")