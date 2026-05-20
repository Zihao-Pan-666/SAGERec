from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from tqdm import tqdm


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def read_jsonl(path: Path) -> pd.DataFrame:
    records = []
    with open_text(path) as fp:
        for line in tqdm(fp, desc=f"Loading {path.name}"):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(records)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix in {".jsonl", ".gz"}:
        return read_jsonl(path)
    raise ValueError(f"Unsupported file format: {path}")


def find_existing_file(data_dir: Path, candidates: Iterable[str]) -> Path:
    for name in candidates:
        path = data_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(
        "Cannot find any of the expected files under "
        f"{data_dir}: {', '.join(candidates)}"
    )


def to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(x) for x in value if x is not None)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def load_amazon_interactions(data_dir: Path, domain: str) -> pd.DataFrame:
    candidates = [
        f"{domain}.csv",
        f"{domain}.jsonl",
        f"{domain}.jsonl.gz",
        f"raw_review_{domain}.jsonl",
        f"raw_review_{domain}.jsonl.gz",
        f"reviews_{domain}.jsonl",
        f"reviews_{domain}.jsonl.gz",
    ]
    path = find_existing_file(data_dir, candidates)
    df = read_table(path)

    rename_map = {}
    if "parent_asin" in df.columns:
        rename_map["parent_asin"] = "ItemId"
    elif "asin" in df.columns:
        rename_map["asin"] = "ItemId"

    if "user_id" in df.columns:
        rename_map["user_id"] = "UserId"
    if "timestamp" in df.columns:
        rename_map["timestamp"] = "Timestamp"

    df = df.rename(columns=rename_map)

    required = ["UserId", "ItemId", "Timestamp"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    df = df[required].dropna()
    df["Timestamp"] = df["Timestamp"].astype(int)
    return df


def load_amazon_metadata(data_dir: Path, domain: str) -> pd.DataFrame:
    candidates = [
        f"meta_{domain}.jsonl",
        f"meta_{domain}.jsonl.gz",
        f"raw_meta_{domain}.jsonl",
        f"raw_meta_{domain}.jsonl.gz",
        f"metadata_{domain}.jsonl",
        f"metadata_{domain}.jsonl.gz",
        f"meta_{domain}.csv",
    ]
    path = find_existing_file(data_dir, candidates)
    meta = read_table(path)

    rename_map = {}
    if "parent_asin" in meta.columns:
        rename_map["parent_asin"] = "ItemId"
    elif "asin" in meta.columns:
        rename_map["asin"] = "ItemId"

    meta = meta.rename(columns=rename_map)
    if "ItemId" not in meta.columns:
        raise ValueError(f"{path} is missing item id column: parent_asin or asin")

    for col in ["title", "description", "features"]:
        if col not in meta.columns:
            meta[col] = ""
        meta[col] = meta[col].apply(to_text)

    return meta[["ItemId", "title", "description", "features"]].dropna(subset=["ItemId"]).drop_duplicates("ItemId")


def kcore_filter(df: pd.DataFrame, k: int) -> pd.DataFrame:
    while True:
        before = len(df)

        item_counts = df["ItemId"].value_counts()
        df = df[df["ItemId"].isin(item_counts[item_counts >= k].index)]

        user_counts = df["UserId"].value_counts()
        df = df[df["UserId"].isin(user_counts[user_counts >= k].index)]

        if len(df) == before:
            return df


def process_domain(domain: str, data_root: str, min_interactions: int, max_seq_len: int) -> dict:
    domain_dir = Path(data_root) / domain

    interactions = load_amazon_interactions(domain_dir, domain)
    metadata = load_amazon_metadata(domain_dir, domain)

    interactions = kcore_filter(interactions, min_interactions)
    interactions = interactions[interactions["ItemId"].isin(metadata["ItemId"])]
    interactions = kcore_filter(interactions, min_interactions)

    data = interactions.merge(metadata, on="ItemId", how="inner")
    data = data.sort_values(["UserId", "Timestamp"]).reset_index(drop=True)
    data = data.drop_duplicates(subset=["UserId", "ItemId"], keep="first")
    data = data.groupby("UserId", group_keys=False).tail(max_seq_len)

    output = domain_dir / "processed_data.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)

    return {
        "domain": domain,
        "n_users": data["UserId"].nunique(),
        "n_items": data["ItemId"].nunique(),
        "n_interactions": len(data),
        "avg_len": data.groupby("UserId").size().mean(),
        "path": str(output),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Amazon Reviews 2023 style domains for SAGERec.")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--domains", type=str, required=True)
    parser.add_argument("--min_interactions", type=int, default=10)
    parser.add_argument("--max_seq_len", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    domains = [x.strip() for x in args.domains.split(",") if x.strip()]
    stats = [process_domain(domain, args.data_root, args.min_interactions, args.max_seq_len) for domain in domains]

    print("\nDataset statistics")
    print(f"{'Domain':<30} {'#Users':>10} {'#Items':>10} {'#Inter':>12} {'AvgLen':>8}")
    print("-" * 76)

    for row in stats:
        print(
            f"{row['domain']:<30} "
            f"{row['n_users']:>10,} "
            f"{row['n_items']:>10,} "
            f"{row['n_interactions']:>12,} "
            f"{row['avg_len']:>8.1f}"
        )


if __name__ == "__main__":
    main()
