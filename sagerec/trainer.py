from __future__ import annotations

import csv
import os
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .losses import unified_alignment_loss


def bpr_loss(pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> torch.Tensor:
    return -torch.log(torch.sigmoid(pos_logits - neg_logits) + 1e-10).mean()


def _sample_training_negatives(
    train_seq: torch.Tensor,
    pos_items: torch.Tensor,
    num_items: int,
    num_negatives: int,
    device: torch.device,
) -> torch.Tensor:
    batch_size = train_seq.size(0)
    neg_items = torch.randint(1, num_items + 1, (batch_size, num_negatives), device=device)
    blocked = torch.cat([train_seq, pos_items.unsqueeze(1)], dim=1)

    for _ in range(16):
        invalid = (neg_items.unsqueeze(-1) == blocked.unsqueeze(1)).any(dim=-1)
        if not invalid.any():
            return neg_items
        neg_items[invalid] = torch.randint(1, num_items + 1, (int(invalid.sum().item()),), device=device)

    invalid = (neg_items.unsqueeze(-1) == blocked.unsqueeze(1)).any(dim=-1)
    for b, n in invalid.nonzero(as_tuple=False).tolist():
        blocked_set = {int(x) for x in blocked[b].tolist() if int(x) != 0}
        while True:
            candidate = int(torch.randint(1, num_items + 1, (1,), device=device).item())
            if candidate not in blocked_set:
                neg_items[b, n] = candidate
                break
    return neg_items


def train_sagerec_model(
    model,
    dataloader,
    optimizer,
    num_epochs: int,
    num_items: int,
    current_domain_id: int,
    num_domains: int,
    device: torch.device,
    save_path: str,
    loss_mode: str = "sage",
    aux_sampler=None,
    train_num_negatives: int = 5,
    lambda_g: float = 0.05,
    gamma_g: float = 0.1,
    beta_id: float = 0.1,
    tau: float = 0.2,
    sim_threshold: float = 0.0,
    patience: int = 10,
    disable_tqdm: bool = False,
) -> str:
    best_loss = float("inf")
    bad_epochs = 0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        model.train()
        totals = {"loss": 0.0, "bpr": 0.0, "gen": 0.0, "sic": 0.0, "div": 0.0, "omega": 0.0}
        start_time = time.time()

        iterator = tqdm(dataloader, desc=f"Epoch {epoch:03d}", leave=False, disable=disable_tqdm)
        for step, batch in enumerate(iterator, start=1):
            seq, pos, _ = batch
            seq = seq.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)

            is_sem = loss_mode == "sem"
            neg = _sample_training_negatives(seq, pos, num_items, train_num_negatives, device)

            logits = model(seq, is_sem_baseline=is_sem)
            pos_logits = logits.gather(1, pos.unsqueeze(1)).repeat(1, train_num_negatives)
            neg_logits = logits.gather(1, neg)
            rec_loss = bpr_loss(pos_logits.reshape(-1), neg_logits.reshape(-1))

            gen_loss = torch.tensor(0.0, device=device, requires_grad=True)
            sic = div = torch.tensor(0.0, device=device)
            omega = torch.tensor(float(lambda_g), device=device)

            if not is_sem and aux_sampler is not None:
                source_items = torch.unique(torch.cat([seq.reshape(-1), pos], dim=0))
                source_items = source_items[source_items > 0]
                aux = aux_sampler.sample(device, size=seq.size(0))

                if source_items.numel() and aux["aux_raw"].numel():
                    source_projected = model.project_items_for_alignment(source_items)
                    source_raw = model.get_raw_item_embeddings(source_items)
                    source_labels = torch.full(
                        (source_items.size(0),),
                        current_domain_id,
                        dtype=torch.long,
                        device=device,
                    )

                    aux_projected = model.project_raw_for_alignment(aux["aux_raw"])
                    all_projected = torch.cat([source_projected, aux_projected], dim=0)
                    all_raw = torch.cat([source_raw, aux["aux_raw"]], dim=0)
                    all_domains = torch.cat([source_labels, aux["aux_domain_ids"]], dim=0)

                    gen_loss, sic, div, omega = unified_alignment_loss(
                        mode=loss_mode,
                        projected_embeddings=all_projected,
                        raw_embeddings=all_raw,
                        domain_labels=all_domains,
                        num_domains=num_domains,
                        source_domain_id=current_domain_id,
                        lambda_g=lambda_g,
                        gamma_g=gamma_g,
                        beta_id=beta_id,
                        tau=tau,
                        sim_threshold=sim_threshold,
                    )

            loss = rec_loss + omega * gen_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            totals["loss"] += float(loss.detach().cpu())
            totals["bpr"] += float(rec_loss.detach().cpu())
            totals["gen"] += float(gen_loss.detach().cpu())
            totals["sic"] += float(sic.detach().cpu())
            totals["div"] += float(div.detach().cpu())
            totals["omega"] += float(omega.detach().cpu())
            iterator.set_postfix({key: f"{value / step:.4f}" for key, value in totals.items()})

        avg_loss = totals["loss"] / max(len(dataloader), 1)
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch:03d} | loss={avg_loss:.4f} | "
            f"bpr={totals['bpr'] / len(dataloader):.4f} | "
            f"gen={totals['gen'] / len(dataloader):.4f} | "
            f"time={elapsed:.1f}s"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            bad_epochs = 0
            torch.save(model.state_dict(), save_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    return save_path


def recall_at_k(pred_items: List[int], ground_truth: int, k: int) -> float:
    return float(ground_truth in pred_items[:k])


def ndcg_at_k(pred_items: List[int], ground_truth: int, k: int) -> float:
    if ground_truth not in pred_items[:k]:
        return 0.0
    rank = pred_items[:k].index(ground_truth) + 1
    return float(1.0 / np.log2(rank + 1))


@torch.no_grad()
def evaluate_model_with_neg_sampling(
    model,
    dataloader,
    top_k_set: List[int],
    num_items: int,
    device: torch.device,
    num_negatives: int = 100,
    is_target_domain: bool = False,
    is_sem_baseline: bool = False,
    seed: int = 2026,
    disable_tqdm: bool = False,
) -> Tuple[Dict[int, float], Dict[int, float], int]:
    rng = random.Random(seed)
    model.eval()

    recall_sum = {k: 0.0 for k in top_k_set}
    ndcg_sum = {k: 0.0 for k in top_k_set}
    total = 0
    item_emb = model.projection_layer(model.pretrained_item_embedding.weight)
    max_k = max(top_k_set)

    iterator = tqdm(dataloader, desc="Eval target" if is_target_domain else "Eval source", leave=False, disable=disable_tqdm)
    for train_seq, val_item, test_item in iterator:
        train_seq = train_seq.to(device)
        val_item = val_item.to(device)
        test_item = test_item.to(device)

        eval_seq = torch.cat([train_seq[:, 1:], val_item.unsqueeze(1)], dim=1)
        history = torch.cat([train_seq, val_item.unsqueeze(1)], dim=1).cpu().numpy()
        test_np = test_item.cpu().numpy()

        candidates = []
        for row, ground_truth in enumerate(test_np):
            blocked = {int(x) for x in history[row] if int(x) != 0}
            blocked.add(int(ground_truth))

            if num_items - len(blocked) < num_negatives:
                raise ValueError("Not enough items to draw evaluation negatives.")

            negs, used = [], set()
            while len(negs) < num_negatives:
                item = rng.randint(1, num_items)
                if item not in blocked and item not in used:
                    negs.append(item)
                    used.add(item)
            candidates.append(negs + [int(ground_truth)])

        candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
        user_rep = model.encode_sequence(eval_seq, is_sem_baseline=is_sem_baseline)
        scores = (user_rep.unsqueeze(1) * item_emb[candidate_tensor]).sum(dim=-1)

        k_eval = min(max_k, candidate_tensor.size(1))
        _, top_indices = torch.topk(scores, k=k_eval, dim=-1)
        top_items = torch.gather(candidate_tensor, 1, top_indices)

        for row, ground_truth in enumerate(test_np):
            pred = top_items[row].tolist()
            for k in top_k_set:
                recall_sum[k] += recall_at_k(pred, int(ground_truth), k)
                ndcg_sum[k] += ndcg_at_k(pred, int(ground_truth), k)
        total += len(test_np)

    return recall_sum, ndcg_sum, total


def metric_dict(recall_sum: Dict[int, float], ndcg_sum: Dict[int, float], total: int, topk: List[int]) -> Dict[str, float]:
    return {
        **{f"R{k}": 100.0 * recall_sum[k] / max(total, 1) for k in topk},
        **{f"N{k}": 100.0 * ndcg_sum[k] / max(total, 1) for k in topk},
    }


def append_results_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    exists = os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
