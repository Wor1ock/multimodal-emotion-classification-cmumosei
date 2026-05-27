from pathlib import Path

import hydra
import joblib
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler

from src.preprocessing.audio import (
    extract_logmel_features,
    extract_mfcc_features,
    load_and_clean_audio,
)
from src.preprocessing.text import (
    clean_texts,
    extract_fasttext_features,
    extract_tfidf_features,
)
from src.utils import make_sample_id, set_seed


def _resolve_csv_path(cfg: DictConfig) -> Path:
    paths = {
        "train": cfg.data.train_csv_path,
        "test": cfg.data.test_csv_path,
        "val": cfg.data.val_csv_path,
    }
    split = cfg.preprocess.split
    if split not in paths:
        raise ValueError(f"Unknown split: {split}")
    return Path(paths[split])


def _extract_single_audio_features(
    row: pd.Series, sample_id: str, audio_dir: Path, cfg: DictConfig
) -> dict:
    audio_path = audio_dir / f"{str(row['video'])}.wav"
    res = {"sample_id": sample_id, "audio_mfcc": None, "audio_logmel": None}

    if not audio_path.exists():
        return res

    waveform, sr = load_and_clean_audio(
        audio_path, row, **cfg.preprocess.audio_core)
    if waveform is None:
        return res

    active_features = cfg.preprocess.features_to_process
    if "audio_mfcc" in active_features:
        res["audio_mfcc"] = extract_mfcc_features(
            waveform, sr, **cfg.preprocess.mfcc)
    if "audio_logmel" in active_features:
        res["audio_logmel"] = extract_logmel_features(
            waveform, sr, **cfg.preprocess.logmel)

    return res


def _scale_and_save_audio_features(
    raw_results: list, feat_key: str, split: str, model_dir: Path, output_dir: Path
) -> None:
    valid_data = [r for r in raw_results if r[feat_key] is not None]
    if not valid_data:
        return

    features = [r[feat_key] for r in valid_data]
    sample_ids = [r["sample_id"] for r in valid_data]

    is_3d = len(features[0].shape) == 2
    flat_features = np.concatenate(
        features, axis=0) if is_3d else np.vstack(features)

    scaler_path = model_dir / f"{feat_key}_scaler.pkl"
    scaler = StandardScaler()

    if split == "train":
        scaler.fit(flat_features)
        joblib.dump(scaler, scaler_path)
    else:
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        scaler = joblib.load(scaler_path)

    save_dir = output_dir / feat_key
    save_dir.mkdir(parents=True, exist_ok=True)

    for sid, feat in zip(sample_ids, features):
        if is_3d:
            T, F = feat.shape
            scaled = scaler.transform(feat).reshape(T, F)
        else:
            scaled = scaler.transform(feat.reshape(1, -1)).squeeze(0)
        torch.save(torch.from_numpy(scaled), save_dir / f"{sid}.pt")


@hydra.main(version_base=None, config_path=".", config_name="config")
def preprocess(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    csv_path = hydra.utils.to_absolute_path(str(_resolve_csv_path(cfg)))
    audio_dir = Path(hydra.utils.to_absolute_path(cfg.preprocess.audio_dir))
    output_dir = Path(hydra.utils.to_absolute_path(
        cfg.preprocess.output_dir)) / cfg.preprocess.split
    model_dir = Path(hydra.utils.to_absolute_path(cfg.preprocess.model_dir))
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df["sample_id"] = [make_sample_id(row) for _, row in df.iterrows()]

    active_features = cfg.preprocess.features_to_process

    if "text_bow" in active_features or "text_embed" in active_features:
        print("Processing text data...")
        cleaned_texts = clean_texts(df["text"].astype(str).tolist())

        if "text_bow" in active_features:
            print("Extracting TF-IDF features...")
            tfidf_gen = extract_tfidf_features(
                cleaned_texts=cleaned_texts,
                split=cfg.preprocess.split,
                model_dir=str(model_dir),
                **cfg.preprocess.tfidf,
            )
            save_dir = output_dir / "text_bow"
            save_dir.mkdir(parents=True, exist_ok=True)
            for sid, tensor in zip(df["sample_id"], tfidf_gen):
                torch.save(tensor, save_dir / f"{sid}.pt")

        if "text_embed" in active_features:
            print("Extracting FastText features...")
            ft_gen = extract_fasttext_features(
                cleaned_texts=cleaned_texts,
                split=cfg.preprocess.split,
                model_dir=str(model_dir),
                **cfg.preprocess.fasttext,
            )
            save_dir = output_dir / "text_embed"
            save_dir.mkdir(parents=True, exist_ok=True)
            for sid, tensor in zip(df["sample_id"], ft_gen):
                torch.save(tensor, save_dir / f"{sid}.pt")

    if "audio_mfcc" in active_features or "audio_logmel" in active_features:
        print("Extracting audio features in parallel...")
        raw_audio_results = joblib.Parallel(n_jobs=-1, backend="loky")(
            joblib.delayed(_extract_single_audio_features)(
                row, row["sample_id"], audio_dir, cfg)
            for _, row in df.iterrows()
        )

        if "audio_mfcc" in active_features:
            print("Scaling and saving MFCC features...")
            _scale_and_save_audio_features(
                raw_audio_results, "audio_mfcc", cfg.preprocess.split, model_dir, output_dir)

        if "audio_logmel" in active_features:
            print("Scaling and saving LogMel features...")
            _scale_and_save_audio_features(
                raw_audio_results, "audio_logmel", cfg.preprocess.split, model_dir, output_dir)


if __name__ == "__main__":
    preprocess()
