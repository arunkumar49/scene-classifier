# Scene Classifier: Development Guide

How this project is developed, trained and fine-tuned on AWS SageMaker AI, with code versioned in GitHub and VS Code as the editor.

## Overview

```
Local VS Code ──(AWS Toolkit, remote)──▶ SageMaker Studio space ◀──git──▶ GitHub
                                                │
                                                ▼ submits
                                    SageMaker Training Jobs ◀──▶ S3 (data, checkpoints, models)
                                                │
                                                ▼ logs
                                    Managed MLflow ──▶ Model Registry ──▶ Endpoint

GitHub Actions (on merge to main, via OIDC) ──▶ SageMaker Pipeline
```

| Where | What happens there |
|---|---|
| Local VS Code | Editor only, connected remotely |
| Studio space (small CPU/GPU) | Writing code, debugging, quick experiments, git |
| SageMaker Training Jobs | All real training and fine-tuning, on temporary GPUs |
| S3 | Data, checkpoints and model artifacts |
| GitHub | Code and configs only |
| Managed MLflow | Experiment tracking |
| Model Registry and Pipelines | Promoting models and automation |

## Ground rules

- **Code and configs go to GitHub. Data and models go to S3.** Never commit datasets, model weights or credentials.
- **Write code on the Studio space; train with Training Jobs.** Don't run long training on the space itself.
- **Every Training Job is tagged with the git commit SHA and the dataset version**, so each model can be traced to the exact code and data that produced it.
- **No long-lived AWS access keys.** Use short-lived credentials on your machine (`aws login` on a personal account, SSO in an organisation) and OIDC in GitHub Actions.
- **Stop what you aren't using.** Studio spaces, MLflow tracking servers and endpoints bill while they run. Training Jobs stop on their own.

## Project structure

```
scene-classifier/
├── artifacts/              # Local scratch outputs (gitignored). Real artifacts live in S3.
├── configs/
│   ├── data/               # Dataset settings: S3 paths, dataset version, splits
│   ├── experiment/         # Combines data + model + training configs for one run
│   ├── model/              # Model architecture settings
│   └── training/           # Hyperparameters, instance type, spot settings
├── data/
│   ├── raw/                # Local samples only (gitignored). Full data lives in S3.
│   └── processed/
├── notebooks/              # Exploration and analysis
├── scripts/                # Entry points, e.g. train.py submits a Training Job
├── serving/                # Inference code for SageMaker endpoints
├── tests/                  # pytest tests
├── src/
│   └── scene_classifier/   # The installable package: reusable code only
│       ├── data/           # Datasets, transforms, dataloaders (code, not data)
│       ├── evaluation/     # Metrics and reports
│       ├── models/         # Model definitions
│       ├── training/       # Training loop (runs inside the Training Job)
│       └── utils/          # Logging, seeding, config loading
├── .gitignore
├── .python-version
├── pyproject.toml
└── README.md
```

Conventions:

- The package folder is `scene_classifier` with an underscore, because that's the import name. `scene-classifier` with a hyphen is only the project name.
- Only `src/scene_classifier/` is packaged. Folders placed directly in `src/` are not importable.
- Each subpackage under `src/scene_classifier/` needs an `__init__.py`.
- `scripts/` and `serving/` stay thin and import from the package, e.g. `from scene_classifier.models import build_model`.
- `tests/` sits outside `src/` so tests run against the installed package.
- `.git` lives at the project root, next to `pyproject.toml`.
- The root `data/` folder holds data files. `src/scene_classifier/data/` holds the code that loads them.
- Git doesn't track empty folders. A folder appears in a commit once it contains a file.

## Step-by-step setup

### 1. AWS prerequisites (platform/admin team)

| Item | Organisation account | Personal account (this project) |
|---|---|---|
| SageMaker AI domain | SSO login (IAM Identity Center), one user profile per developer | **IAM authentication** via "Set up for single user (Quick setup)". Identity Center access to AWS accounts needs AWS Organizations, which upgrades a Free plan and forfeits its credits. |
| Execution role | Least-privilege role per team | The role Quick setup creates, plus MLflow permissions |
| S3 bucket | e.g. `s3://<org>-ml-dev/scene-classifier/` | One bucket, e.g. `s3://scene-classifier-<suffix>/` |
| Experiment tracking | Managed MLflow | A serverless **MLflow App** (no extra charge; you pay for S3 storage). The older MLflow tracking servers bill per hour. |
| GPU capacity | Already approved | **Request service quota increases** for GPU training instances (new accounts often start at 0) |
| Budget | Central FinOps | AWS Budgets alert on the account |

Remote access from VS Code is switched on **per space** (the "Remote access" toggle, or by the AWS Toolkit when you connect). The identity on your local machine needs `sagemaker:StartSession` and related permissions; an admin user already has them.

### 2. Local machine

1. Install VS Code 1.90 or later, AWS CLI v2 (2.32 or later for `aws login`), and the **Session Manager plugin** for the AWS CLI.
2. Sign in with short-lived credentials: `aws login --profile <name>` (personal account) or `aws configure sso` (organisation).
3. Install the **AWS Toolkit** and **Remote - SSH** extensions in VS Code. On Windows, OpenSSH and PowerShell 5.1 or later are also needed.
4. Sign in to the AWS Toolkit with the same profile.

### 3. Connect VS Code to SageMaker

1. In the AWS Toolkit explorer, go to SageMaker AI → Studio → your space → **Connect**.
2. A remote VS Code window opens. Its terminal and Python run on the space.

Notes:

- The space needs at least 8 GB of memory, e.g. ml.t3.large. ml.t3.medium is not supported for remote connections.
- A GPU space (e.g. ml.g5.xlarge) needs its own Studio service quota and bills per hour; use it only for short GPU debugging.
- Space storage defaults to 5 GB, which is too small for Python environments, dataset caches and model weights. Set it to 25–50 GB (maximum 100 GB).
- The space is billed while it runs. Stop it when idle, and turn on idle shutdown in the domain settings.

### 4. GitHub on the Studio space

1. In the remote terminal, run `gh auth login` or add an SSH key to GitHub.
2. Clone the repo into `/home/sagemaker-user/`. This folder survives space restarts, but is deleted if the space itself is deleted.
3. Work on feature branches, open pull requests for review, and merge into a protected `main`.

### 5. Data in S3

```
s3://<bucket>/scene-classifier/
├── data/raw/
├── data/processed/v1/
├── data/processed/v2/
├── checkpoints/
└── models/
```

- Each dataset version gets a new prefix (`v1/`, `v2/`, ...). Never overwrite an existing version.
- The local `data/` and `artifacts/` folders are scratch space only and are gitignored.
- Heavy preprocessing runs as a SageMaker Processing Job.

### 6. Training and fine-tuning

- `scripts/train.py` is the launcher. It submits a Training Job with the SageMaker Python SDK (`ModelTrainer`). SDK V3 moved the import paths (`from sagemaker.train import ModelTrainer`); V2 examples online use different ones, so match the docs to the installed major version.
- The job runs the `src/scene_classifier/` code in an AWS PyTorch Deep Learning Container on a GPU instance, e.g. ml.g4dn.xlarge or ml.g5.xlarge. The instance type needs an approved service quota (separate quotas for on-demand and spot training).
- Input is a dataset prefix in S3, and the trained model is written back to S3.
- Hyperparameters come from `configs/`.
- Turn on managed spot training and save checkpoints to S3. This costs up to 90% less than on-demand.
- Tag each job with the git commit SHA and the dataset version.

### 7. Experiment tracking

- From the training code, log params, metrics, the git commit SHA and the dataset version to the SageMaker MLflow App.
- The execution role needs the `sagemaker-mlflow` permissions to log to it.
- Compare runs in the MLflow UI, opened from Studio.

### 8. Model Registry and automation

- Register the best model in the SageMaker Model Registry, where it moves from pending to approved.
- A SageMaker Pipeline processes the data, trains, evaluates, and registers the model only if it meets a metric threshold.
- GitHub Actions runs on each merge to `main`. It assumes an AWS role through OIDC (no stored keys) and starts the pipeline.

### 9. Deployment

- `serving/` code is deployed to a SageMaker endpoint from an approved model.
- A real-time endpoint runs (and bills) continuously until deleted. On a personal account, delete it after testing, or use a serverless endpoint (CPU only, scales to zero).
- Larger organisations use separate AWS accounts for dev, staging and prod. A personal account keeps one account and separates environments by naming.

## Fixes before the first Training Job

### Python version

`pyproject.toml` currently requires Python `>=3.14`. The AWS PyTorch containers use an older Python (3.12 at the time of writing), so the package won't install inside a Training Job.

- In `pyproject.toml`, set `requires-python = ">=3.11"`.
- Set `.python-version` to `3.12`.

### .gitignore

Add these entries. The leading `/` matters: plain `data/` would also ignore `src/scene_classifier/data/` and `configs/data/`.

```
/data/
/artifacts/
.venv/
__pycache__/
.ipynb_checkpoints/
```
