import hashlib
import os

import lightning as L
import matplotlib.pyplot as plt
import pandas as pd
import torch


def set_seed(seed: int = 42) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)

    L.seed_everything(seed, workers=True)

    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_sample_id(row: pd.Series) -> str:
    if "id" in row.index and pd.notna(row["id"]):
        return str(row["id"])

    video = str(row["video"]).strip()
    start = str(row["start_time"]).strip()
    end = str(row["end_time"]).strip()

    payload = f"{video}_{start}_{end}"

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def plot_metrics_from_log(log_path: str) -> None:
    metrics = pd.read_csv(log_path)

    metrics_epoch = metrics.groupby("epoch").mean().reset_index()
    epochs = metrics_epoch["epoch"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss
    train_loss = metrics_epoch.get("train_loss")
    val_loss = metrics_epoch.get("val_loss")

    if train_loss is not None:
        ax1.plot(epochs, train_loss, "b-", label="Training Loss", linewidth=2)
    if val_loss is not None:
        ax1.plot(epochs, val_loss, "r-", label="Valid Loss", linewidth=2)
    ax1.set_title("Loss")
    ax1.set_xlabel("Epochs")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # F1 Score
    train_f1 = metrics_epoch.get("train_f1")
    val_f1 = metrics_epoch.get("val_f1")

    if train_f1 is not None:
        ax2.plot(epochs, train_f1, "b-", label="Training F1", linewidth=2)
    if val_f1 is not None:
        ax2.plot(epochs, val_f1, "r-", label="Valid F1", linewidth=2)
    ax2.set_title("F1 Score")
    ax2.set_xlabel("Epochs")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
