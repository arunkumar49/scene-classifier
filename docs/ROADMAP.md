# Scene Classifier: Implementation Roadmap

High-level steps to build the scene classifier end to end: fine-tune and compare ResNet-50 and ViT-Base on the Intel Image Classification dataset (6 classes), track experiments in MLflow, and serve the best model with FastAPI on AWS.

Status: **Done**, **In progress**, **To do**.

## Summary

| # | Step | Week | Status |
|---|---|---|---|
| 1 | Project skeleton | 0 | Done |
| 2 | Load data, split and upload to S3 | 0 | Done (loose ends) |
| 3 | Load model | 1 | In progress |
| 4 | Model, training and experiment configs | 1 | To do |
| 5 | Transforms | 1 | To do |
| 6 | Lightning DataModule | 1 | To do |
| 7 | LightningModule | 1 | To do |
| 8 | Training script and local smoke test | 1 | To do |
| 9 | MLflow tracking | 2 | To do |
| 10 | Add ViT-Base | 2 | To do |
| 11 | SageMaker Training Jobs | 2 | To do |
| 12 | Hyperparameter runs | 2 | To do |
| 13 | Evaluation module | 3 | To do |
| 14 | Robustness and error analysis | 3 | To do |
| 15 | Model selection and registry | 3 | To do |
| 16 | FastAPI inference service | 4 | To do |
| 17 | Docker image | 4 | To do |
| 18 | Deploy on AWS | 4 | To do |
| 19 | Monitoring and CI | 4 | To do |
| 20 | Documentation | 4 | To do |

## Week 0: Foundation

### 1. Project skeleton (Done)

Package layout, `pyproject.toml` with uv, pre-commit hooks and the [Development Guide](DEVELOPMENT.md).

### 2. Load data, split and upload to S3 (Done, loose ends)

[`LoadData`](../src/scene_classifier/data/data_cli.py) loads the Hub dataset, carves a validation split from train, and checks the labels against the fixed classes in the [data config](../conf/data/hf_intel-image-classification.yaml). [`upload_dataset_to_s3`](../src/scene_classifier/data/s3_upload.py) uploads each split.

Loose ends:

- The validation split is not stratified yet. `split_train_test_val` calls `train_test_split` without `stratify_by_column`, but the config requires a stratified split (DR-01).
- The S3 key hardcodes `data/raw/` and ignores `prefix` and `dataset_version` from the config.
- S3 class folders are named by class ID (`0` to `5`) instead of class name.

## Week 1: ResNet-50 baseline

### 3. Load model (In progress)

- Done: [`get_model`](../src/scene_classifier/models/resnet.py) loads `microsoft/resnet-50` with a 6-class head and the class names in `id2label` / `label2id`.
- Left: a freezing option (train the head only, or fine-tune the whole network).

### 4. Model, training and experiment configs

- YAML files in `conf/model/`, `conf/training/` and `conf/experiment/`, each with a frozen-dataclass loader like `utils/data_config.py`.
- An experiment config combines one data, model and training config for a run.

### 5. Transforms

- Resize the 150 × 150 images to 224 × 224 and apply ImageNet normalisation, taking the values from each model's `AutoImageProcessor`.
- Train-only augmentation: horizontal flip, small rotation, colour jitter.
- Validation, test and serving use the same transforms without augmentation. Write them once so serving can reuse them.

### 6. Lightning DataModule

- Train, validation and test DataLoaders with configurable batch size, number of workers and augmentation (FR-02).
- Reads from the Hub locally and from S3 inside a Training Job.

### 7. LightningModule

- Forward pass, training, validation and test steps, and the loss.
- Optimiser and learning-rate scheduler from config.
- Freezing strategy from config.
- Metrics with torchmetrics: accuracy and macro-F1.

### 8. Training script and local smoke test

- `scripts/train.py`, with seeds set for every random number generator (FR-08).
- Early stopping, and best-checkpoint selection on validation macro-F1 (FR-05).
- Mixed-precision training on GPU, with CPU fallback (FR-06).
- Smoke test: 1 epoch on a small subset, on CPU.

**Milestone:** one clean ResNet-50 baseline.

## Week 2: Tracking and comparison

### 9. MLflow tracking

Log params, metrics, the git commit SHA, the dataset version and checkpoints as artifacts for every run.

### 10. Add ViT-Base

A generic `build_model()` driven by the model config (`AutoModelForImageClassification`), so switching between ResNet-50 and `google/vit-base-patch16-224` is a config change only (FR-04).

### 11. SageMaker Training Jobs

- A launcher that submits the job with the dataset's S3 prefix as input.
- Managed spot training, with checkpoints saved to S3.
- Each job tagged with the git commit SHA and the dataset version.

### 12. Hyperparameter runs

- Search at least learning rate × freezing strategy, with 6 or more runs per model (FR-07).
- Compare the ResNet-50 and ViT-Base runs in the MLflow UI.

**Milestone:** both models trained and compared.

## Week 3: Evaluation

### 13. Evaluation module

- On the test set: per-class precision, recall and F1, macro-F1, and a confusion matrix.
- Reports saved as MLflow artifacts.

### 14. Robustness and error analysis

- Rotated and blurred variants of the test set.
- The most confused class pairs and examples of misclassified images.
- Grad-CAM for ResNet-50 and attention maps for ViT-Base.

### 15. Model selection and registry

- Weigh accuracy against latency and running cost.
- Register the best model in the MLflow Model Registry.
- Choose the confidence threshold below which a prediction is flagged for human review (`needs_review`).

**Milestone:** best model chosen.

## Week 4: Serving and deployment

### 16. FastAPI inference service

- Endpoints: `POST /v1/predict`, `POST /v1/predict/batch`, `GET /v1/model-info`, `GET /health`.
- Each prediction returns the label, class ID, confidence, top-3 classes and the `needs_review` flag.
- Uses the same preprocessing as step 5.

### 17. Docker image

Build the service image and test it locally.

### 18. Deploy on AWS

A SageMaker endpoint or EC2. Delete the endpoint when it isn't in use, because it bills while running.

### 19. Monitoring and CI

- Structured JSON logs, plus a CloudWatch dashboard and alerts.
- GitHub Actions running lint, tests and the CPU smoke test.

### 20. Documentation

README, model card, and the ResNet-50 vs ViT-Base comparison report.

## Notes

- Write tests alongside each step rather than at the end. Steps 5, 6 and 7 are the most worth testing.
- Steps 4 and 5 come next, because the DataModule and LightningModule depend on them.
