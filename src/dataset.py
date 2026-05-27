from pathlib import Path
from typing import List

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils import make_sample_id


class MoseiDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        features_to_load: List[str],
        split_dir: Path | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.features_to_load = features_to_load
        self.split_dir = split_dir
        self.target_cols = ["happy", "sad",
                            "anger", "surprise", "disgust", "fear"]
        self._sample_ids = [make_sample_id(row)
                            for _, row in self.df.iterrows()]

    def __len__(self) -> int:
        return len(self.df)

    def _load_feature(self, sample_id: str, key: str) -> torch.Tensor:
        return torch.load(self.split_dir / key / f"{sample_id}.pt")

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        sample_id = self._sample_ids[idx]

        labels = torch.tensor(
            row[self.target_cols].values.astype(np.float32),
            dtype=torch.float32,
        )

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
        features_to_process: List[str],
        batch_size: int,
        num_workers: int,
        **kwargs,
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
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.datasets["train"], shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.datasets["val"], shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.datasets["test"], shuffle=False)
