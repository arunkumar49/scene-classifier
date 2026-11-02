from typing import Any
import torch
import torch.nn as nn
import lightning.pytorch as L
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import optim
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from transformers import PreTrainedModel

class ClassifierModule(L.LightningModule):
    
    def __init__(
        self, 
        model: PreTrainedModel,
        learning_rate: float,
        weight_decay: float = 0.0,
        freeze_backbone: bool = False
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()

        # from_pretrained returns the model in eval mode, and Lightning keeps that mode
        self.model.train()
        
        # Train only the classification head: freeze the pretrained backbone
        if freeze_backbone:
            for param in self.model.base_model.parameters():
                param.requires_grad = False
            # Eval mode also stops BatchNorm from updating its running statistics
            self.model.base_model.eval()

        # One copy per stage: metrics accumulate over an epoch, so stages can't share them
        num_classes = model.config.num_labels
        if num_classes is None:
            raise ValueError("The model config has no id2label, so the number of classes is unknown.")
        metrics = MetricCollection({
            # average="micro" is plain accuracy; torchmetrics' default ("macro") is balanced accuracy
            "acc": MulticlassAccuracy(num_classes=num_classes, average="micro"),
            "macro_f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
        })
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        
    
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # HF models return an output object; the steps only need the logits
        return self.model(pixel_values=pixel_values).logits
    
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Run one training batch: compute the loss and update the train metrics.
        """
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.train_metrics.update(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        # Lightning computes the metrics over the whole epoch, then resets them
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)
        
        return loss
        
        
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """
        Run one validation batch: compute the loss and update the validation metrics.
        """
        
        x, y = batch
        
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        self.val_metrics.update(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        # Lightning computes the metrics over the whole epoch, then resets them
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)
        
    
    
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """
        Run one test batch: compute the loss and update the test metrics.
        """
                
        x, y = batch
                
        logits = self(x)
        loss = self.loss_fn(logits, y)
                
        self.test_metrics.update(logits, y)
        self.log("test_loss", loss)
        # Lightning computes the metrics over the whole epoch, then resets them
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)
        
        
        
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
        )
        
        return optimizer