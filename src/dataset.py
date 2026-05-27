from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.preprocessing import FEATURE_KEYS
from src.preprocessing.sample_id import make_sample_id


class MoseiDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        features_dir: str | None = None,
        is_train: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.features_dir = Path(features_dir) if features_dir else None
        self.is_train = is_train
        self.target_cols = ["happy", "sad", "anger", "surprise", "disgust", "fear"]
        self._sample_ids = [make_sample_id(self.df.iloc[i]) for i in range(len(self.df))]

    def __len__(self) -> int:
        return len(self.df)

    def _load_feature(self, sample_id: str, key: str) -> torch.Tensor:
        if self.features_dir is None:
            raise ValueError("features_dir is required to load precomputed tensors")
        path = self.features_dir / key / f"{sample_id}.pt"
        return torch.load(path, weights_only=True)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.df.iloc[idx]
        sample_id = self._sample_ids[idx]
        labels = torch.tensor(
            row[self.target_cols].values.astype("float32"),
            dtype=torch.float32,
        )

        sample: dict[str, torch.Tensor | str] = {
            "text": str(row["text"]),
            "video_id": str(row["video"]),
            "sample_id": sample_id,
            "labels": labels,
        }

        if self.features_dir is not None:
            for key in FEATURE_KEYS:
                sample[key] = self._load_feature(sample_id, key)

        return sample


class MoseiDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_csv_path: str,
        test_csv_path: str,
        batch_size: int,
        num_workers: int,
        val_size: float,
        seed: int,
        features_dir: str | None = None,
    ):
        super().__init__()
        self.train_csv_path = Path(train_csv_path)
        self.test_csv_path = Path(test_csv_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.seed = seed
        self.features_dir = features_dir

        self.train_df: pd.DataFrame | None = None
        self.val_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None
        self.train_ds: MoseiDataset | None = None
        self.val_ds: MoseiDataset | None = None
        self.test_ds: MoseiDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            full_train_df = pd.read_csv(self.train_csv_path)
            unique_videos = full_train_df["video"].unique()

            g = np.random.default_rng(self.seed)
            g.shuffle(unique_videos)

            val_count = int(len(unique_videos) * self.val_size)
            val_videos = unique_videos[:val_count]

            self.val_df = full_train_df[full_train_df["video"].isin(val_videos)]
            self.train_df = full_train_df[~full_train_df["video"].isin(val_videos)]

            self.train_ds = MoseiDataset(
                self.train_df,
                features_dir=self.features_dir,
                is_train=True,
            )
            self.val_ds = MoseiDataset(
                self.val_df,
                features_dir=self.features_dir,
                is_train=False,
            )

        if stage == "test" or stage is None:
            self.test_df = pd.read_csv(self.test_csv_path)
            self.test_ds = MoseiDataset(
                self.test_df,
                features_dir=self.features_dir,
                is_train=False,
            )

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
        if self.train_ds is None:
            raise RuntimeError("Call setup('fit') before train_dataloader")
        return self._make_dataloader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None:
            raise RuntimeError("Call setup('fit') before val_dataloader")
        return self._make_dataloader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_ds is None:
            raise RuntimeError("Call setup('test') before test_dataloader")
        return self._make_dataloader(self.test_ds, shuffle=False)
