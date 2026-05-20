from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import load_pretrained_embeddings, remap_processed_data_item_ids, resolve_embedding_path


class SequenceDataset(Dataset):
    def __init__(self, data: pd.DataFrame, max_seq_length: int, dataset_name: str):
        required = {"UserId", "ItemId", "Timestamp"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"{dataset_name} is missing columns: {sorted(missing)}")

        df = data.copy()
        df["UserId"] = df["UserId"].astype(str)
        df, self.raw_to_model_map = remap_processed_data_item_ids(df, dataset_name)
        df = df.sort_values(["UserId", "Timestamp"]).reset_index(drop=True)

        self.dataset_name = dataset_name
        self.max_seq_length = max_seq_length
        self.data_frame = df
        self.num_items = int(df["ItemId"].max())
        self.user_sequences = self._build_sequences()

        if int(df["ItemId"].nunique()) != self.num_items:
            raise ValueError(f"{dataset_name}: remapped ItemId must be dense in 1..N.")

    def _build_sequences(self) -> List[List[int]]:
        sequences = []
        for _, group in self.data_frame.groupby("UserId", sort=False):
            seq = group["ItemId"].astype(int).tolist()
            if len(seq) >= 3:
                sequences.append(seq)
        return sequences

    def __len__(self) -> int:
        return len(self.user_sequences)

    def get_num_items(self) -> int:
        return self.num_items

    def __getitem__(self, index: int):
        sequence = self.user_sequences[index]
        train_items = sequence[:-2]
        val_item = sequence[-2]
        test_item = sequence[-1]

        train_seq = np.zeros(self.max_seq_length, dtype=np.int64)
        length = min(len(train_items), self.max_seq_length)
        if length:
            train_seq[-length:] = np.asarray(train_items[-length:], dtype=np.int64)

        return (
            torch.tensor(train_seq, dtype=torch.long),
            torch.tensor(val_item, dtype=torch.long),
            torch.tensor(test_item, dtype=torch.long),
        )


def load_sequence_dataset(domain: str, data_root: str, max_seq_length: int) -> SequenceDataset:
    path = f"{data_root}/{domain}/processed_data.csv"
    data = pd.read_csv(path)
    return SequenceDataset(data, max_seq_length=max_seq_length, dataset_name=domain)


def load_domain_semantic_embeddings(
    domain: str,
    data_root: str = "./data",
    embedding_tag: str = "llama",
    embedding_path_template: str = "",
) -> torch.Tensor:
    path = resolve_embedding_path(domain, data_root, embedding_tag, embedding_path_template)
    return load_pretrained_embeddings(path)


class AuxiliarySemanticSampler:
    def __init__(
        self,
        aux_domains: List[str],
        all_domains: List[str],
        data_root: str = "./data",
        embedding_tag: str = "llama",
        embedding_path_template: str = "",
        sample_mode: str = "domain_uniform",
        batch_size: int = 1024,
    ):
        if sample_mode not in {"domain_uniform", "global_uniform"}:
            raise ValueError("sample_mode must be domain_uniform or global_uniform.")
        if not aux_domains:
            raise ValueError("At least one auxiliary domain is required.")

        self.sample_mode = sample_mode
        self.batch_size = batch_size
        self.domain_cache = []

        for domain in aux_domains:
            tensor = load_domain_semantic_embeddings(domain, data_root, embedding_tag, embedding_path_template)
            raw = tensor[1:].cpu()
            domain_id = all_domains.index(domain)
            self.domain_cache.append({"name": domain, "raw": raw, "domain_id": domain_id})

        if sample_mode == "global_uniform":
            self.global_raw = torch.cat([pack["raw"] for pack in self.domain_cache], dim=0)
            self.global_domains = torch.cat([
                torch.full((pack["raw"].size(0),), pack["domain_id"], dtype=torch.long)
                for pack in self.domain_cache
            ])

    @staticmethod
    def _sample_indices(n_total: int, k: int) -> torch.Tensor:
        if n_total <= k:
            return torch.arange(n_total, dtype=torch.long)
        return torch.randint(0, n_total, (k,), dtype=torch.long)

    @torch.no_grad()
    def sample(self, device: torch.device, size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        k = size or self.batch_size

        if self.sample_mode == "domain_uniform":
            pack = self.domain_cache[np.random.randint(0, len(self.domain_cache))]
            indices = self._sample_indices(pack["raw"].size(0), k)
            raw = pack["raw"][indices].to(device)
            domains = torch.full((raw.size(0),), pack["domain_id"], dtype=torch.long, device=device)
            return {"aux_raw": raw, "aux_domain_ids": domains}

        indices = self._sample_indices(self.global_raw.size(0), k)
        return {
            "aux_raw": self.global_raw[indices].to(device),
            "aux_domain_ids": self.global_domains[indices].to(device),
        }
