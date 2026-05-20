from __future__ import annotations

import argparse
import ast
import html
import math
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, List

import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from llm2vec import LLM2Vec
except ImportError as exc:
    LLM2Vec = None
    LLM2VEC_IMPORT_ERROR = exc


MISSING_STRINGS = {"", "nan", "none", "null", "na", "n/a", "[]", "{}"}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""

    text = html.unescape(str(value))
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\\n", " ").replace("\\t", " ")
    text = re.sub(r"\\u[0-9a-fA-F]{4}", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    return text


def parse_listish(value: Any) -> List[str]:
    raw = normalize_text(value)
    if raw.lower() in MISSING_STRINGS:
        return []

    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple)):
                return [normalize_text(x) for x in parsed if normalize_text(x)]
        except Exception:
            raw = raw[1:-1].strip()

    return [] if raw.lower() in MISSING_STRINGS else [raw]


def first_existing(row: pd.Series, columns: List[str]) -> str:
    for col in columns:
        if col in row.index:
            value = normalize_text(row[col])
            if value:
                return value
    return ""


def join_field(items: List[str], fallback: str) -> str:
    seen, output = set(), []
    for item in items:
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            output.append(item)
    return "; ".join(output) if output else fallback


def amazon_prompt(row: pd.Series) -> str:
    title = first_existing(row, ["title", "Title", "product_title", "products"]) or "no title provided"
    description = join_field(parse_listish(first_existing(row, ["description", "Description", "desc", "text"])), "no description provided")
    features = join_field(parse_listish(first_existing(row, ["features", "feature", "Feature", "categories"])), "no feature provided")
    return f"Please summarize the following item: title: {title}. Feature: {features}. Description: {description}"


def steam_prompt(row: pd.Series) -> str:
    title = first_existing(row, ["products", "title", "Title"]) or "no title provided"
    review = first_existing(row, ["text", "description", "review"]) or "no review text provided"
    return f"Please summarize the following game item: title: {title}. Review text: {review}"


def build_item_table(domain: str, data_root: str, max_items: int = 0) -> pd.DataFrame:
    path = Path(data_root) / domain / "processed_data.csv"
    df = pd.read_csv(path).drop_duplicates("ItemId").reset_index(drop=True)
    df = df.head(max_items) if max_items else df

    if "steam" in domain.lower():
        df["prompt"] = df.apply(steam_prompt, axis=1)
        df["ItemId"] = range(len(df))
    else:
        df["prompt"] = df.apply(amazon_prompt, axis=1)
        df["ItemId"] = range(1, len(df) + 1)
    return df[["ItemId", "prompt"]]


def encode_domain(args, domain: str) -> None:
    if LLM2Vec is None:
        raise ImportError("llm2vec is not installed.") from LLM2VEC_IMPORT_ERROR

    output = Path(args.data_root) / domain / f"{domain}_embedding_{args.embedding_tag}.parquet"
    if output.exists() and not args.overwrite:
        print(f"{domain}: skip existing {output}")
        return

    item_table = build_item_table(domain, args.data_root, args.max_items_per_domain)
    model = LLM2Vec.from_pretrained(args.model_name, device_map=args.device)

    embeddings = []
    start = time.time()
    for start_idx in tqdm(range(0, len(item_table), args.batch_size), desc=f"Encoding {domain}"):
        batch = item_table["prompt"].iloc[start_idx : start_idx + args.batch_size].tolist()
        emb = model.encode(batch)
        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().numpy()
        embeddings.extend(emb.tolist())

    result = pd.DataFrame({
        "ItemId": item_table["ItemId"].astype(int),
        "item_text_embedding": embeddings,
    })
    result.to_parquet(output, index=False)
    print(f"{domain}: saved {len(result)} embeddings to {output} in {time.time() - start:.1f}s")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--domains", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--embedding_tag", type=str, default="llama")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_items_per_domain", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    for domain in [x.strip() for x in args.domains.split(",") if x.strip()]:
        encode_domain(args, domain)


if __name__ == "__main__":
    main()
