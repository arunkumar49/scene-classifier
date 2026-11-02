"""Submit a SageMaker Training Job that runs scripts/train.py on a GPU.

Run from the project root; extra arguments are Hydra overrides for the job:
    uv run --env-file .env.sagemaker --group launch python scripts/launch_job.py training=full_finetune
"""

import os
import subprocess
import sys
import tarfile
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

import boto3
from omegaconf import OmegaConf
from sagemaker.core import image_uris
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.shapes import Tag
from sagemaker.core.training.configs import (
    Compute,
    OutputDataConfig,
    SourceCode,
    StoppingCondition,
)
from sagemaker.train.model_trainer import ModelTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Bucket, region, AWS profile and dataset version are read from here, not repeated
DATA_CONFIG = PROJECT_ROOT / "conf" / "data" / "hf_intel-image-classification.yaml"

INSTANCE_TYPE = "ml.g5.xlarge"  # 1x A10G GPU; the account quota is 1
# Must match the torch~=2.9.0 pin, so `pip install .` keeps the image's torch
IMAGE_VERSION = "2.9.0"
VOLUME_SIZE_GB = 50
MAX_RUNTIME_SECONDS = 6 * 3600
BASE_JOB_NAME = "scene-classifier"
# The job role can only read and write under this prefix. Outputs go to
# <prefix>/<job>/output; the SDK stages the code under the output path's prefix,
# so <prefix>/staging/<job>/input
JOBS_PREFIX = "scene-classifier/jobs"

# Always true for a Training Job; the command line adds the rest
JOB_OVERRIDES = [
    "trainer=gpu",
    "tracking=sagemaker",
    "data.s3.profile=null",  # no AWS profile in the container; boto3 uses the job role
    "hydra.run.dir=/opt/ml/output/data/run",  # SageMaker uploads this folder to S3
]

# Left out of the code archive. Matched against names at every level, so never
# add "data": it would also drop src/scene_classifier/data/ and conf/data/
IGNORE_PATTERNS = [
    ".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache",
    "outputs", "artifacts", "processed", "raw", "notebooks",
    "mlruns", "mlflow.db", ".env*", "*.ckpt",
]


def require_env(name: str) -> str:
    """Return an environment variable, or stop with a hint about the env file."""
    value = os.environ.get(name)
    if not value:
        sys.exit(f"{name} is not set; run with --env-file .env.sagemaker")
    return value


def git_sha() -> str:
    """Return HEAD's commit SHA, refusing to launch from uncommitted code."""
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, cwd=PROJECT_ROOT
        ).stdout.strip()

    if git("status", "--porcelain"):
        sys.exit("Commit your changes first: the job is tagged with the commit it runs")
    return git("rev-parse", "HEAD")


def pack_source(dest_dir: Path) -> Path:
    """Tar the project for the job, leaving out IGNORE_PATTERNS.

    The SDK uploads a source folder from Windows with backslashes in nested S3 keys,
    so src/scene_classifier/ never reaches the container. tarfile always writes "/".
    """
    def keep(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
        name = PurePosixPath(member.name).name
        return None if any(fnmatch(name, pattern) for pattern in IGNORE_PATTERNS) else member

    archive = dest_dir / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(PROJECT_ROOT, arcname=".", filter=keep)
    return archive


def main() -> None:
    s3 = OmegaConf.load(DATA_CONFIG).s3
    sha = git_sha()

    # The SDK's role pre-check demands ec2 network-interface permissions, which AWS
    # documents as needed only for jobs in a VPC; this job doesn't use one
    os.environ.setdefault("SAGEMAKER_VALIDATE_ROLE", "false")

    session = Session(
        boto_session=boto3.Session(profile_name=s3.profile, region_name=s3.region),
        default_bucket=s3.bucket,
    )
    image = image_uris.retrieve(
        framework="pytorch",
        region=s3.region,
        version=IMAGE_VERSION,
        py_version="py312",
        instance_type=INSTANCE_TYPE,
        image_scope="training",
    )
    overrides = " ".join(JOB_OVERRIDES + sys.argv[1:])

    # The archive must exist until train() has uploaded it
    with TemporaryDirectory() as tmp:
        trainer = ModelTrainer(
            sagemaker_session=session,
            role=require_env("SAGEMAKER_ROLE_ARN"),
            training_image=image,
            base_job_name=BASE_JOB_NAME,
            input_s3_key_prefix="staging",
            source_code=SourceCode(
                # The container runs `tar -xzf` on it before the command
                source_dir=str(pack_source(Path(tmp))),
                # pip, not uv: the lock's exact torch build would replace the image's
                command=(
                    "pip install . sagemaker-mlflow==0.5.0 && "
                    f"python scripts/train.py {overrides}"
                ),
            ),
            compute=Compute(
                instance_type=INSTANCE_TYPE, instance_count=1, volume_size_in_gb=VOLUME_SIZE_GB
            ),
            stopping_condition=StoppingCondition(max_runtime_in_seconds=MAX_RUNTIME_SECONDS),
            output_data_config=OutputDataConfig(s3_output_path=f"s3://{s3.bucket}/{JOBS_PREFIX}"),
            # Only these two: AWS_PROFILE would break boto3 in the container
            environment={
                "MLFLOW_TRACKING_URI": require_env("MLFLOW_TRACKING_URI"),
                "GIT_COMMIT_SHA": sha,
            },
            tags=[
                Tag(key="git_sha", value=sha),
                Tag(key="dataset_version", value=s3.dataset_version),
            ],
        )
        # Streams the job's logs here. Ctrl+C stops the streaming, not the job
        trainer.train(wait=True)


if __name__ == "__main__":
    main()
