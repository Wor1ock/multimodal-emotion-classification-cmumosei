import os
import re
from pathlib import Path
from typing import Generator, List

import fasttext
import joblib
import numpy as np
import torch
from num2words import num2words
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing.utils_text import *


def text2num(match: re.Match) -> str:
    val = match.group()
    if len(val) > 15:
        return val
    try:
        return num2words(val, lang="en")
    except (ValueError, TypeError):
        return val


def split_mixed_numchar(match: re.Match) -> str:
    word = match.group()
    num = "".join(c for c in word if c.isdigit())
    txt = "".join(c for c in word if c.isalpha())
    try:
        num_word = num2words(num, lang="en") if num else ""
    except (ValueError, TypeError):
        num_word = num
    return f"{num_word} {txt}".strip()


def pre_regex_clean(text: str) -> str:
    text = text.lower()
    text = RE_SUFFIX.sub(r"\1", text)
    text = RE_MIXED.sub(split_mixed_numchar, text)
    return RE_NUMBER.sub(text2num, text)


def clean_texts(texts: List[str]) -> List[List[str]]:
    cleaned_regex = [pre_regex_clean(t) for t in texts]
    cleaned_docs = []

    for doc in nlp.pipe(cleaned_regex, n_process=-1, batch_size=256):
        tokens = [
            t.lemma_.lower() for t in doc
            if t.is_alpha and len(t) > 1 and
            (t.text.lower() not in stop_words or t.text.lower() in custom_exceptions)
        ]
        cleaned_docs.append(tokens)

    return cleaned_docs


def extract_tfidf_features(
    cleaned_texts: List[List[str]],
    split: str,
    model_dir: str,
    max_features: int,
    ngram_range: List[int],
    min_df: int
) -> Generator[torch.Tensor, None, None]:
    text_strings = [" ".join(tokens) for tokens in cleaned_texts]
    vectorizer_path = Path(model_dir) / "tfidf_vectorizer.pkl"
    vectorizer_path.parent.mkdir(parents=True, exist_ok=True)

    if split == "train":
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=tuple(ngram_range),
            min_df=min_df
        )
        tfidf_matrix = vectorizer.fit_transform(text_strings)
        joblib.dump(vectorizer, vectorizer_path)
    else:
        if not vectorizer_path.exists():
            raise FileNotFoundError(
                f"Vectorizer not found at {vectorizer_path}.")
        vectorizer = joblib.load(vectorizer_path)
        tfidf_matrix = vectorizer.transform(text_strings)

    for i in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix[i].toarray().astype(np.float32).squeeze(0)
        yield torch.from_numpy(row)


def extract_fasttext_features(
    cleaned_texts: List[List[str]],
    split: str,
    model_dir: str,
    pretrained_path: str,
    epoch: int,
    lr: float,
    dim: int
) -> Generator[torch.Tensor, None, None]:
    model_dir_path = Path(model_dir)
    ft_model_path = model_dir_path / "fasttext_finetuned.bin"
    ft_model_path.parent.mkdir(parents=True, exist_ok=True)

    if split == "train":
        tmp_train_file = model_dir_path / "tmp_train_data.txt"
        with open(tmp_train_file, "w", encoding="utf-8") as f:
            for tokens in cleaned_texts:
                f.write(" ".join(tokens) + "\n")

        model = fasttext.train_unsupervised(
            input=str(tmp_train_file),
            model="skipgram",
            pretrainedVectors=pretrained_path if os.path.exists(
                pretrained_path) else "",
            epoch=epoch,
            lr=lr,
            dim=dim
        )
        model.save_model(str(ft_model_path))
        if tmp_train_file.exists():
            tmp_train_file.unlink()
    else:
        if not ft_model_path.exists():
            raise FileNotFoundError(
                f"FastText model not found at {ft_model_path}.")
        model = fasttext.load_model(str(ft_model_path))

    model_dim = model.get_dimension()
    for tokens in cleaned_texts:
        vectors = [model.get_word_vector(t) for t in tokens]
        if not vectors:
            avg_vector = np.zeros(model_dim, dtype=np.float32)
        else:
            avg_vector = np.mean(vectors, axis=0).astype(np.float32)
        yield torch.from_numpy(avg_vector)
