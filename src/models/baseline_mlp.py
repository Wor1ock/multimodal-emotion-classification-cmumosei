from typing import Any, Literal

import lightning as L
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch import Tensor
from torch.optim import AdamW


class FeatureMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MLPBaselineModel(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        learning_rate: float,
        text_bow_dim: int,
        text_embed_dim: int,
        audio_mfcc_dim: int,
        audio_logmel_dim: int,
        hidden_dim: int,
        dropout: float,
        f1_average: Literal["micro", "macro"] = "micro",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.f1_average = f1_average
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.text_bow_mlp = FeatureMLP(text_bow_dim, hidden_dim, num_classes, dropout)
        self.text_embed_mlp = FeatureMLP(text_embed_dim, hidden_dim, num_classes, dropout)
        self.audio_mfcc_mlp = FeatureMLP(audio_mfcc_dim, hidden_dim, num_classes, dropout)
        self.audio_logmel_mlp = FeatureMLP(audio_logmel_dim, hidden_dim, num_classes, dropout)

        self._train_logits: list[Tensor] = []
        self._train_targets: list[Tensor] = []
        self._val_logits: list[Tensor] = []
        self._val_targets: list[Tensor] = []

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        logits = [
            self.text_bow_mlp(batch["text_bow"]),
            self.text_embed_mlp(batch["text_embed"]),
            self.audio_mfcc_mlp(batch["audio_mfcc"]),
            self.audio_logmel_mlp(batch["audio_logmel"]),
        ]
        return torch.stack(logits, dim=0).mean(dim=0)

    def _shared_step(self, batch: dict[str, Any], stage: Literal["train", "val"]) -> Tensor:
        logits = self(batch)
        loss = self.loss_fn(logits, batch["labels"])
        storage_logits = self._train_logits if stage == "train" else self._val_logits
        storage_targets = self._train_targets if stage == "train" else self._val_targets
        storage_logits.append(logits.detach())
        storage_targets.append(batch["labels"].detach())
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "val")

    def _epoch_f1(
        self,
        logits_list: list[Tensor],
        targets_list: list[Tensor],
        prefix: str,
    ) -> None:
        if not logits_list:
            return
        logits = torch.cat(logits_list, dim=0)
        targets = torch.cat(targets_list, dim=0)
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
        targets_np = targets.cpu().numpy()
        micro = f1_score(targets_np, preds, average="micro", zero_division=0)
        macro = f1_score(targets_np, preds, average="macro", zero_division=0)
        self.log(f"{prefix}_f1", micro if self.f1_average == "micro" else macro, prog_bar=True)
        self.log(f"{prefix}_f1_micro", micro)
        self.log(f"{prefix}_f1_macro", macro)

    def on_train_epoch_end(self) -> None:
        self._epoch_f1(self._train_logits, self._train_targets, "train")
        self._train_logits.clear()
        self._train_targets.clear()

    def on_validation_epoch_end(self) -> None:
        self._epoch_f1(self._val_logits, self._val_targets, "val")
        self._val_logits.clear()
        self._val_targets.clear()

    def configure_optimizers(self) -> AdamW:
        return AdamW(self.parameters(), lr=self.learning_rate)
