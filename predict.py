from pathlib import Path

import hydra
import lightning as L
import pandas as pd
import torch
from omegaconf import DictConfig

from src.dataset import MoseiDataModule
from src.utils import set_seed


@hydra.main(version_base=None, config_path=".", config_name="config")
def predict(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    dm = MoseiDataModule(
        **cfg.data,
    )
    dm.setup(stage="test")

    checkpoint_path = cfg.predict.checkpoint_path
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = hydra.utils.instantiate(cfg.model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
    )

    predictions = trainer.predict(model, dataloaders=dm.test_dataloader())

    all_logits = torch.cat(list(predictions), dim=0)
    all_probs = torch.sigmoid(all_logits)
    all_preds = (all_probs > 0.5).long()

    target_cols = ["happy", "sad", "anger", "surprise", "disgust", "fear"]

    df = pd.DataFrame(all_preds.numpy(), columns=target_cols)
    df["probs"] = [p.tolist() for p in all_probs]

    test_dataset = dm.datasets["test"]
    df["sample_id"] = [test_dataset._sample_ids[i] for i in range(len(df))]
    df["text"] = [test_dataset.df.iloc[i]["text"] for i in range(len(df))]

    output_path = Path(cfg.predict.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    predict()
