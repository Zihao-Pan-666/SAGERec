from __future__ import annotations

import ast
import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_domains(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def safe_float_str(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def resolve_embedding_path(
    domain: str,
    data_root: str = "./data",
    embedding_tag: str = "llama",
    embedding_path_template: str = "",
) -> str:
    if embedding_path_template:
        path = embedding_path_template.format(
            data_root=data_root,
            domain=domain,
            tag=embedding_tag,
        )
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"Embedding file not found: {path}")

    candidates = [
        os.path.join(data_root, domain, f"{domain}_embedding_{embedding_tag}.parquet"),
    ]
    if embedding_tag == "llama":
        candidates.append(os.path.join(data_root, domain, f"{domain}_embedding_llama3.parquet"))

    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Cannot find embedding parquet. Tried: " + ", ".join(candidates))


def _parse_embedding_cell(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float32)
    if isinstance(value, str):
        return np.asarray(ast.literal_eval(value), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def load_pretrained_embeddings(parquet_path: str) -> torch.Tensor:
    df = pd.read_parquet(parquet_path)
    required = {"ItemId", "item_text_embedding"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{parquet_path} is missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"{parquet_path} is empty.")
    if df["ItemId"].isna().any() or df["ItemId"].duplicated().any():
        raise ValueError(f"{parquet_path} contains invalid ItemId values.")

    df = df.copy().sort_values("ItemId").reset_index(drop=True)
    item_ids = df["ItemId"].astype(int)
    min_id, max_id = int(item_ids.min()), int(item_ids.max())
    if min_id not in (0, 1):
        raise ValueError(f"Unexpected ItemId range in {parquet_path}: starts from {min_id}")

    zero_based = min_id == 0
    embeddings = np.stack(df["item_text_embedding"].apply(_parse_embedding_cell).values)
    tensor = torch.zeros((max_id + 2 if zero_based else max_id + 1, embeddings.shape[1]), dtype=torch.float32)

    for item_id, emb in zip(item_ids, embeddings):
        model_id = int(item_id) + 1 if zero_based else int(item_id)
        tensor[model_id] = torch.tensor(emb, dtype=torch.float32)

    tensor[0] = 0.0
    return tensor


def _build_steam_id_map(data: pd.DataFrame) -> Dict[Any, int]:
    return {raw_id: idx + 1 for idx, raw_id in enumerate(data["ItemId"].unique())}


def _build_amazon_id_map(data: pd.DataFrame) -> Dict[Any, int]:
    stage1 = {raw_id: idx for idx, raw_id in enumerate(data["ItemId"].unique())}
    temp_ids = data["ItemId"].map(stage1)
    stage2 = {old_id: idx for idx, old_id in enumerate(temp_ids.unique(), start=1)}
    return {raw_id: stage2[stage1[raw_id]] for raw_id in stage1}


def build_raw_to_model_item_id_map(data: pd.DataFrame, dataset_name: str) -> Dict[Any, int]:
    if "ItemId" not in data.columns:
        raise ValueError("Input dataframe must contain ItemId.")
    if "steam" in dataset_name.lower():
        return _build_steam_id_map(data)
    return _build_amazon_id_map(data)


def remap_processed_data_item_ids(
    data: pd.DataFrame,
    dataset_name: str,
    item_col: str = "ItemId",
) -> Tuple[pd.DataFrame, Dict[Any, int]]:
    if item_col not in data.columns:
        raise ValueError(f"Input dataframe must contain {item_col}.")

    df = data.copy()
    mapping = build_raw_to_model_item_id_map(df.rename(columns={item_col: "ItemId"}), dataset_name)
    df["RawItemId"] = df[item_col]
    df[item_col] = df[item_col].map(mapping)

    missing = int(df[item_col].isna().sum())
    if missing:
        raise ValueError(f"{dataset_name}: failed to remap {missing} item ids.")
    df[item_col] = df[item_col].astype(int)
    return df, mapping
