from pathlib import Path

import hydra
import lightning as L
import pandas as pd
import torch
from omegaconf import DictConfig

from src.dataset import MoseiDataModule
from src.utils import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="predict")
def predict(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    dm = MoseiDataModule(**cfg.data)
    dm.setup(stage="test")

    checkpoint_path = Path(cfg.predict.checkpoint_path)
    if not checkpoint_path.exists():
        checkpoint_dir = checkpoint_path.parent
        ckpt_files = list(checkpoint_dir.glob("*.ckpt"))
        ckpt_files.sort(key=lambda x: x.stat().st_mtime)
        checkpoint_path = ckpt_files[-1]

    model = hydra.utils.instantiate(cfg.model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
    )

    batch_outputs = trainer.predict(model, dataloaders=dm.test_dataloader())

    all_logits = torch.cat([batch["logits"] for batch in batch_outputs], dim=0)

    all_sample_ids = []
    all_texts = []
    for batch in batch_outputs:
        all_sample_ids.extend(batch["sample_id"])
        all_texts.extend(batch["text"])

    all_probs = torch.sigmoid(all_logits)
    all_preds = (all_probs > 0.5).long()

    target_cols = ["happy", "sad", "anger", "surprise", "disgust", "fear"]
    df = pd.DataFrame(all_preds.numpy(), columns=target_cols)
    df["probs"] = [p.tolist() for p in all_probs]
    df["sample_id"] = all_sample_ids
    df["text"] = all_texts

    output_path = Path(cfg.predict.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    predict()
