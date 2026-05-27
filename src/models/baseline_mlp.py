from typing import Any, Dict
import lightning as L
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torchmetrics.classification import MultilabelF1Score


class MLPBaselineModel(L.LightningModule):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        lr: float,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = lr

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.f1_metric = MultilabelF1Score(
            num_labels=num_classes,
            average="macro"
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.model(x)

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        x, y = batch["x"], batch["labels"]

        logits = self(x)
        loss = self.loss_fn(logits, y.float())

        preds = torch.sigmoid(logits)
        self.f1_metric(preds, y.long())

        self.log("train_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("train_f1", self.f1_metric, on_step=False,
                 on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        x, y = batch["x"], batch["labels"]

        logits = self(x)
        loss = self.loss_fn(logits, y.float())

        preds = torch.sigmoid(logits)
        self.f1_metric(preds, y.long())

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.f1_metric, on_step=False,
                 on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> AdamW:
        return AdamW(self.parameters(), lr=self.learning_rate)
