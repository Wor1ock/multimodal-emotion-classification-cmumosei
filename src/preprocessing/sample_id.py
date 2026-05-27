import hashlib

import pandas as pd


def make_sample_id(row: pd.Series) -> str:
    if "id" in row.index and pd.notna(row["id"]):
        return str(row["id"])
    payload = f"{row['video']}|{row['text']}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
