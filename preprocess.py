from pathlib import Path

import hydra
import joblib
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler

from src.preprocessing import FEATURE_KEYS
from src.preprocessing.audio import *
from src.preprocessing.sample_id import make_sample_id
from src.preprocessing.text import *
from src.utils import set_seed


def _resolve_csv_path(cfg: DictConfig) -> Path:
    if cfg.preprocess.split == "train":
        return Path(cfg.data.train_csv_path)
    if cfg.preprocess.split == "test":
        return Path(cfg.data.test_csv_path)
    if cfg.preprocess.split == "val":
        return Path(cfg.data.val_csv_path)
    raise ValueError(f"Unknown split: {cfg.preprocess.split}")


def _resolve_audio_path(audio_dir: Path, row: pd.Series) -> Path:
    return audio_dir / f"{str(row['video'])}.wav"


def _save_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)


@hydra.main(version_base=None, config_path=".", config_name="config")
def preprocess(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    csv_path = hydra.utils.to_absolute_path(str(_resolve_csv_path(cfg)))
    audio_dir = Path(hydra.utils.to_absolute_path(cfg.preprocess.audio_dir))
    base_out = hydra.utils.to_absolute_path(cfg.preprocess.output_dir)
    output_dir = Path(base_out) / cfg.preprocess.split
    model_dir = Path(hydra.utils.to_absolute_path(cfg.preprocess.model_dir))
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    sample_ids = [make_sample_id(row) for _, row in df.iterrows()]
    raw_texts = df["text"].astype(str).tolist()

    print("Extracting text features...")
    cleaned_texts = clean_texts(raw_texts)

    if "text_bow" in FEATURE_KEYS:
        text_bows = extract_tfidf_features(
            cleaned_texts=cleaned_texts,
            split=cfg.preprocess.split,
            model_dir=str(model_dir),
            **cfg.preprocess.tfidf
        )
        joblib.Parallel(n_jobs=-1, backend="threading")(
            joblib.delayed(_save_tensor)(
                output_dir / "text_bow" / f"{sid}.pt", tensor)
            for sid, tensor in zip(sample_ids, text_bows)
        )

    if "text_embed" in FEATURE_KEYS:
        text_embeds = extract_fasttext_features(
            cleaned_texts=cleaned_texts,
            split=cfg.preprocess.split,
            model_dir=str(model_dir),
            **cfg.preprocess.fasttext
        )
        joblib.Parallel(n_jobs=-1, backend="threading")(
            joblib.delayed(_save_tensor)(
                output_dir / "text_embed" / f"{sid}.pt", tensor)
            for sid, tensor in zip(sample_ids, text_embeds)
        )

    print("Extracting audio features...")
    mfcc_list = []
    mel_list = []
    audio_sample_ids = []

    for sample_id, (_, row) in zip(sample_ids, df.iterrows()):
        audio_path = _resolve_audio_path(audio_dir, row)

        if not audio_path.exists():
            print(f"Warning: File missing {audio_path}")
            continue

        waveform, sr = load_and_clean_audio(
            audio_path, row, **cfg.preprocess.audio_core)
        if waveform is None:
            continue

        audio_sample_ids.append(sample_id)

        if "audio_mfcc" in FEATURE_KEYS:
            mfcc = extract_mfcc_features(waveform, sr, **cfg.preprocess.mfcc)
            mfcc_list.append(mfcc)

        if "audio_logmel" in FEATURE_KEYS:
            logmel = extract_logmel_features(
                waveform, sr, **cfg.preprocess.logmel)
            mel_list.append(logmel)

    if "audio_mfcc" in FEATURE_KEYS and mfcc_list:
        mfcc_matrix = np.stack(mfcc_list)
        scaler_path = model_dir / "mfcc_scaler.pkl"

        if cfg.preprocess.split == "train":
            scaler = StandardScaler()
            mfcc_scaled = scaler.fit_transform(mfcc_matrix)
            joblib.dump(scaler, scaler_path)
        else:
            if not scaler_path.exists():
                raise FileNotFoundError(f"Scaler not found at {scaler_path}")
            scaler = joblib.load(scaler_path)
            mfcc_scaled = scaler.transform(mfcc_matrix)

        joblib.Parallel(n_jobs=-1, backend="threading")(
            joblib.delayed(_save_tensor)(
                output_dir / "audio_mfcc" / f"{sid}.pt", torch.from_numpy(arr))
            for sid, arr in zip(audio_sample_ids, mfcc_scaled)
        )

    if "audio_logmel" in FEATURE_KEYS and mel_list:
        mel_matrix = np.stack(mel_list)
        scaler_path = model_dir / "mel_scaler.pkl"

        if cfg.preprocess.split == "train":
            scaler = StandardScaler()
            mel_scaled = scaler.fit_transform(mel_matrix)
            joblib.dump(scaler, scaler_path)
        else:
            if not scaler_path.exists():
                raise FileNotFoundError(f"Scaler not found at {scaler_path}")
            scaler = joblib.load(scaler_path)
            mel_scaled = scaler.transform(mel_matrix)

        joblib.Parallel(n_jobs=-1, backend="threading")(
            joblib.delayed(_save_tensor)(
                output_dir / "audio_logmel" / f"{sid}.pt", torch.from_numpy(arr))
            for sid, arr in zip(audio_sample_ids, mel_scaled)
        )


if __name__ == "__main__":
    preprocess()
