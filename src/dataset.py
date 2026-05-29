from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils import make_sample_id


def _collate_fn(batch: list[dict]) -> dict:
    result = {}
    first_item = batch[0]
    keys = first_item.keys()

    for key in ("text", "video_id", "sample_id"):
        if key in keys:
            result[key] = [b[key] for b in batch]

    if "labels" in keys:
        result["labels"] = torch.stack([b["labels"] for b in batch])

    feature_keys = [k for k in keys if k not in ("text", "video_id", "sample_id", "labels")]

    if feature_keys:
        active_key = feature_keys[0]
        values = [b[active_key] for b in batch]

        if values[0].ndim == 1:
            result["x"] = torch.stack(values)
        else:
            from torch.nn.utils.rnn import pad_sequence

            result["x"] = pad_sequence(values, batch_first=True, padding_value=0.0).float()
            result["x_lengths"] = torch.tensor([v.size(0) for v in values], dtype=torch.long)

    return result


class MoseiDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        features_to_load: list[str],
        split_dir: Path,
    ):
        self.df = df.reset_index(drop=True)
        self.features_to_load = features_to_load
        self.split_dir = split_dir
        self.target_cols = ["happy", "sad", "anger", "surprise", "disgust", "fear"]

        self._sample_ids = self.df.apply(make_sample_id, axis=1).tolist()

    def __len__(self) -> int:
        return len(self.df)

    def _load_feature(self, sample_id: str, key: str) -> torch.Tensor:
        path = self.split_dir / key / f"{sample_id}.pt"

        if not path.exists():
            if key == "text_bow":
                return torch.zeros(300, dtype=torch.float32)
            if key == "text_embed":
                return torch.zeros(768, dtype=torch.float32)
            if key == "audio_mfcc":
                return torch.zeros(1, 128, dtype=torch.float32)
            if key == "audio_logmel":
                return torch.zeros(1, 256, dtype=torch.float32)
            return torch.zeros(1, dtype=torch.float32)

        return torch.load(path).float()

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        sample_id = self._sample_ids[idx]

        raw_values = row[self.target_cols].values.astype(np.float32)
        binary_values = (raw_values > 0.0).astype(np.float32)
        labels = torch.tensor(binary_values, dtype=torch.float32)

        sample = {
            "text": str(row["text"]),
            "video_id": str(row["video"]),
            "sample_id": sample_id,
            "labels": labels,
        }

        for key in self.features_to_load:
            sample[key] = self._load_feature(sample_id, key)

        return sample


class MoseiDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_csv_path: str,
        val_csv_path: str,
        test_csv_path: str,
        features_dir: str,
        features_to_process: list[str],
        batch_size: int,
        num_workers: int,
        **_kwargs,
    ):
        super().__init__()
        self.csv_paths = {
            "train": Path(train_csv_path),
            "val": Path(val_csv_path),
            "test": Path(test_csv_path),
        }
        self.features_base_dir = Path(features_dir)
        self.features_to_load = features_to_process
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.datasets = {"train": None, "val": None, "test": None}

    def _make_split(self, split: str) -> MoseiDataset:
        return MoseiDataset(
            df=pd.read_csv(self.csv_paths[split]),
            features_to_load=self.features_to_load,
            split_dir=self.features_base_dir / split,
        )

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.datasets["train"] = self._make_split("train")
            self.datasets["val"] = self._make_split("val")

        if stage == "test" or stage is None:
            self.datasets["test"] = self._make_split("test")

    def _make_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=_collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.datasets["train"], shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.datasets["val"], shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.datasets["test"], shuffle=False)
