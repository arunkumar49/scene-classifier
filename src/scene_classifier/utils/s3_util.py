import mimetypes
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import boto3
from botocore.config import Config
from datasets import Dataset, Image

from scene_classifier.utils.data_config import S3UploadConfig

if TYPE_CHECKING:
    # boto3-stubs is a dev dependency, so only import it for the type checker
    from mypy_boto3_s3 import S3Client
    from mypy_boto3_s3.type_defs import PutObjectRequestTypeDef


def get_s3_client(
    profile: str | None = None, region: str | None = None, max_workers: int = 16
) -> "S3Client":
    """Create an S3 client for an AWS profile. profile=None uses the default AWS chain
    (env vars, ~/.aws/credentials, or an IAM role)."""
    return boto3.Session(profile_name=profile).client(
        "s3",
        region_name=region,
        config=Config(
            max_pool_connections=max_workers,
            retries={"max_attempts": 5, "mode": "adaptive"},
        ),
    )
    
    
    
def list_s3_images(client, bucket: str, prefix: str, image_suffixes: list[str]) -> Iterator[tuple[str, int]]:
    """Yield (key, size) for every image under the prefix. Handles pagination
    (list_objects_v2 returns at most 1000 keys per call)."""
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(tuple(image_suffixes)):
                yield key, obj["Size"]
                
                
                
def download_split(
    bucket: str,
    s3_prefix: str,
    split: str,
    local_dir: str | Path,
    image_suffixes: list[str],
    max_workers: int = 16,
    region: str | None = None,
    profile: str | None = None,
) -> Path:
    """
    Mirror s3://{bucket}/{s3_prefix}/{split}/{class}/{file}
    into   {local_dir}/{split}/{class}/{file}

    Files that already exist locally with the same size are skipped,
    so re-running is cheap.
    """
    client = get_s3_client(profile=profile, region=region, max_workers=max_workers)
    split_prefix = f"{s3_prefix.strip('/')}/{split}/"
    dest_root = Path(local_dir) / split

    to_download: list[tuple[str, Path]] = []
    for key, size in list_s3_images(client, bucket, split_prefix, image_suffixes):
        relative = key[len(split_prefix):]          # e.g. "0/20057.jpg"
        dest = dest_root / relative
        if dest.exists() and dest.stat().st_size == size:
            continue
        to_download.append((key, dest))

    if not to_download:
        # logger.info("Split '%s' already up to date at %s", split, dest_root)
        return dest_root

    # logger.info("Downloading %d files for split '%s'", len(to_download), split)

    def _download(key: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        client.download_file(bucket, key, str(tmp))
        tmp.replace(dest)  # atomic: no half-written files if interrupted

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_download, k, d): k for k, d in to_download}
        for fut in as_completed(futures):
            fut.result()  # re-raise any download error

    return dest_root




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
        s3_key = f"data/{split}/{item['label']}/{image_name}"

        # Upload to S3
        request: PutObjectRequestTypeDef = {
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
