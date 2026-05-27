from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import pandas as pd


def load_and_clean_audio(
    audio_path: str | Path,
    row: pd.Series,
    target_sr: int = 22050,
    res_type: str = "kaiser_fast"
) -> Tuple[Optional[np.ndarray], int]:
    path_str = str(Path(audio_path))
    start = float(row["start_time"])
    duration = float(row["end_time"]) - start

    y, sr = librosa.load(
        path_str,
        sr=target_sr,
        offset=start,
        duration=duration,
        res_type=res_type
    )

    y, _ = librosa.effects.trim(y)

    if len(y) < 1024:
        return None, sr

    max_val = np.max(np.abs(y))
    y = y / (max_val + 1e-9)

    y = np.append(y[0], y[1:] - 0.97 * y[:-1])

    return y, sr


def extract_mfcc_features(
    waveform: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 20
) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=n_mfcc)
    mfcc_feat = np.hstack([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])
    return mfcc_feat.astype("float32")


def extract_logmel_features(
    waveform: np.ndarray,
    sample_rate: int,
    n_mels: int = 128
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sample_rate, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_feat = np.hstack([np.mean(mel_db, axis=1), np.std(mel_db, axis=1)])
    return mel_feat.astype("float32")
