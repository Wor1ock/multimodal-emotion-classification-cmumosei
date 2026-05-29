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

        self.train_f1 = MultilabelF1Score(num_labels=num_classes, average="macro")
        self.val_f1 = MultilabelF1Score(num_labels=num_classes, average="macro")

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.model(x)

    def training_step(self, batch: dict[str, Tensor], _batch_idx: int) -> Tensor:
        x, y = batch["x"], batch["labels"]

        logits = self(x)
        loss = self.loss_fn(logits, y.float())

        self.train_f1.update(logits, y.long())

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train_f1", self.train_f1.compute(), prog_bar=True)
        self.train_f1.reset()

    def validation_step(self, batch: dict[str, Tensor], _batch_idx: int) -> Tensor:
        x, y = batch["x"], batch["labels"]

        logits = self(x)
        loss = self.loss_fn(logits, y.float())

        self.val_f1.update(logits, y.long())

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        if self.val_f1.tp is not None:
            self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        self.val_f1.reset()

    def predict_step(self, batch: dict[str, torch.Tensor], _batch_idx: int) -> dict:
        x = batch["x"]
        logits = self(x)

        return {"logits": logits, "sample_id": batch["sample_id"], "text": batch["text"]}

    def configure_optimizers(self) -> AdamW:
        return AdamW(self.parameters(), lr=self.learning_rate)
