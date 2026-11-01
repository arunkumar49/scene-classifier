from collections.abc import Mapping
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig

from scene_classifier.data.data_module import DataModule
from scene_classifier.data.transform import (
    build_eval_transforms,
    build_train_transforms,
)
from scene_classifier.models.vision_model import build_model
from scene_classifier.training.classifier_module import ClassifierModule
from scene_classifier.utils.data_config import to_data_config
from scene_classifier.utils.model_config import to_model_config
from scene_classifier.utils.trainer_config import to_trainer_config
from scene_classifier.utils.training_config import to_training_config

# Early stopping and the best checkpoint both follow validation macro-F1 (FR-05)
MONITOR = "val_macro_f1"


def train(config: DictConfig, output_dir: str | Path) -> Mapping[str, float]:
    """
    Fine-tune the model, then return the test metrics of the best checkpoint.
    """
    # 1. Turn each config section into a typed, frozen dataclass
    model_config = to_model_config(config.model)
    data_config = to_data_config(config.data)
    trainer_config = to_trainer_config(config.trainer)
    training_config = to_training_config(config.training)

    # 2. Use the typed values from here on.
    # Seed first: build_model gives the new head random weights
    L.seed_everything(training_config.seed, workers=True)

    vision_model = build_model(model_config, data_config.classes)
    model = ClassifierModule(
        vision_model,
        training_config.learning_rate,
        training_config.weight_decay,
        training_config.freeze_backbone,
    )

    data_module = DataModule(
        data_config,
        build_train_transforms(model_config),
        build_eval_transforms(model_config),
    )

    trainer = L.Trainer(
        accelerator=trainer_config.accelerator,
        devices=trainer_config.devices,
        precision=trainer_config.precision,
        limit_train_batches=trainer_config.limit_train_batches,
        limit_val_batches=trainer_config.limit_val_batches,
        limit_test_batches=trainer_config.limit_test_batches,
        max_epochs=training_config.max_epochs,
        default_root_dir=output_dir,
        # Set explicitly: Lightning's default becomes TensorBoard once it's installed
        logger=CSVLogger(save_dir=output_dir),
        callbacks=[
            EarlyStopping(
                monitor=MONITOR, mode="max", patience=training_config.patience
            ),
            ModelCheckpoint(
                monitor=MONITOR, mode="max", dirpath=Path(output_dir) / "checkpoints"
            ),
        ],
    )

    # With datamodule=, Lightning runs prepare_data and setup before using the loaders
    trainer.fit(model, datamodule=data_module)

    # "best" reloads the checkpoint with the highest val_macro_f1, not the last epoch
    return trainer.test(model, datamodule=data_module, ckpt_path="best")[0]
