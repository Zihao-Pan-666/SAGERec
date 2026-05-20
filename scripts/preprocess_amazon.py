from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def load_amazon_interactions(data_dir: Path, domain: str) -> pd.DataFrame:
    path = data_dir / f"{domain}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Interaction file not found: {path}")

    df = pd.read_csv(path)
    if "parent_asin" in df.columns:
        df = df.rename(columns={"parent_asin": "ItemId"})
    elif "asin" in df.columns:
        df = df.rename(columns={"asin": "ItemId"})

    df = df.rename(columns={"user_id": "UserId", "timestamp": "Timestamp"})
    df = df[["UserId", "ItemId", "Timestamp"]].dropna()
    df["Timestamp"] = df["Timestamp"].astype(int)
    return df


def load_amazon_metadata(data_dir: Path, domain: str) -> pd.DataFrame:
    path = data_dir / f"meta_{domain}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    records = []
    with path.open("r", encoding="utf-8") as fp:
        for line in tqdm(fp, desc=f"Loading {path.name}"):
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    meta = pd.DataFrame(records)
    if "parent_asin" in meta.columns:
        meta = meta.rename(columns={"parent_asin": "ItemId"})
    elif "asin" in meta.columns:
        meta = meta.rename(columns={"asin": "ItemId"})

    for col in ["title", "description", "features"]:
        if col not in meta.columns:
            meta[col] = ""
        meta[col] = meta[col].apply(lambda x: str(x) if isinstance(x, list) else x)

    return meta[["ItemId", "title", "description", "features"]].drop_duplicates("ItemId")


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
    data = data.groupby("UserId", group_keys=False).apply(lambda g: g.tail(max_seq_len))

    output = domain_dir / "processed_data.csv"
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
    parser = argparse.ArgumentParser(description="Preprocess Amazon-style domains for SAGERec.")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--domains", type=str, required=True)
    parser.add_argument("--min_interactions", type=int, default=10)
    parser.add_argument("--max_seq_len", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    domains = [x.strip() for x in args.domains.split(",") if x.strip()]
    stats = [process_domain(d, args.data_root, args.min_interactions, args.max_seq_len) for d in domains]

    print("\nDataset statistics")
    print(f"{'Domain':<30} {'#Users':>10} {'#Items':>10} {'#Inter':>12} {'AvgLen':>8}")
    print("-" * 76)
    for row in stats:
        print(
            f"{row['domain']:<30} {row['n_users']:>10,} {row['n_items']:>10,} "
            f"{row['n_interactions']:>12,} {row['avg_len']:>8.1f}"
        )


if __name__ == "__main__":
    main()
