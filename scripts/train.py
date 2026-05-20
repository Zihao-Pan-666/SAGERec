from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sagerec.datasets import AuxiliarySemanticSampler, load_sequence_dataset
from sagerec.models import build_model
from sagerec.trainer import (
    append_results_csv,
    evaluate_model_with_neg_sampling,
    metric_dict,
    train_sagerec_model,
)
from sagerec.utils import load_pretrained_embeddings, parse_domains, resolve_embedding_path, safe_float_str, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAGERec on a source domain and evaluate zero-shot transfer.")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--source_domain", type=str, required=True)
    parser.add_argument("--all_domains", type=str, required=True)
    parser.add_argument("--aux_domains", type=str, default="")
    parser.add_argument("--target_domains", type=str, default="")
    parser.add_argument("--embedding_tag", type=str, default="llama")
    parser.add_argument("--embedding_path_template", type=str, default="")

    parser.add_argument("--model", type=str, default="bert4rec", choices=["sasrec", "bert4rec", "gru4rec", "unisrec"])
    parser.add_argument("--loss_mode", type=str, default="sage", choices=["sem", "recg", "sage"])
    parser.add_argument("--hidden_units", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--n_exps", type=int, default=8, help="Number of MoE experts for UniSRec.")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--lambda_g", type=float, default=0.05)
    parser.add_argument("--gamma_g", type=float, default=0.1)
    parser.add_argument("--beta_id", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--sim_threshold", type=float, default=0.0)

    parser.add_argument("--aux_samples", type=int, default=4096)
    parser.add_argument("--aux_sample_mode", type=str, default="domain_uniform", choices=["domain_uniform", "global_uniform"])
    parser.add_argument("--train_negatives", type=int, default=5)
    parser.add_argument("--eval_negatives", type=int, default=100)
    parser.add_argument("--topk", type=str, default="10,20")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--disable_tqdm", action="store_true")
    return parser.parse_args()


def init_model(args, source_embeddings, device):
    return build_model(
        model_name=args.model,
        hidden_units=args.hidden_units,
        max_seq_length=args.max_len,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate,
        pretrained_item_embeddings=source_embeddings,
        n_exps=args.n_exps,
    ).to(device)


def run_name(args):
    return "_".join([
        args.model,
        f"src_{args.source_domain}",
        args.loss_mode,
        f"lam{safe_float_str(args.lambda_g)}",
        f"gam{safe_float_str(args.gamma_g)}",
        f"beta{safe_float_str(args.beta_id)}",
        f"tau{safe_float_str(args.tau)}",
        f"seed{args.seed}",
    ])


def main():
    args = parse_args()
    set_seed(args.seed)

    all_domains = parse_domains(args.all_domains)
    if args.source_domain not in all_domains:
        raise ValueError("--source_domain must appear in --all_domains.")

    aux_domains = parse_domains(args.aux_domains) or [d for d in all_domains if d != args.source_domain]
    target_domains = parse_domains(args.target_domains) or aux_domains
    topk = [int(x) for x in parse_domains(args.topk)]

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    source_dataset = load_sequence_dataset(args.source_domain, args.data_root, args.max_len)
    source_embedding_path = resolve_embedding_path(
        args.source_domain,
        args.data_root,
        args.embedding_tag,
        args.embedding_path_template,
    )
    source_embeddings = load_pretrained_embeddings(source_embedding_path).to(device)

    train_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True)
    model = init_model(args, source_embeddings, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    name = run_name(args)
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    checkpoint_path = args.checkpoint or os.path.join(checkpoint_dir, f"{name}.pth")

    aux_sampler = None
    if args.loss_mode != "sem":
        aux_sampler = AuxiliarySemanticSampler(
            aux_domains=aux_domains,
            all_domains=all_domains,
            data_root=args.data_root,
            embedding_tag=args.embedding_tag,
            embedding_path_template=args.embedding_path_template,
            sample_mode=args.aux_sample_mode,
            batch_size=args.aux_samples,
        )

    if not args.eval_only:
        train_sagerec_model(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            num_epochs=args.epochs,
            num_items=source_dataset.get_num_items(),
            current_domain_id=all_domains.index(args.source_domain),
            num_domains=len(all_domains),
            device=device,
            save_path=checkpoint_path,
            loss_mode=args.loss_mode,
            aux_sampler=aux_sampler,
            train_num_negatives=args.train_negatives,
            lambda_g=args.lambda_g,
            gamma_g=args.gamma_g,
            beta_id=args.beta_id,
            tau=args.tau,
            sim_threshold=args.sim_threshold,
            patience=args.patience,
            disable_tqdm=args.disable_tqdm,
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    rows = []

    for target in target_domains:
        target_dataset = load_sequence_dataset(target, args.data_root, args.max_len)
        target_embedding_path = resolve_embedding_path(
            target,
            args.data_root,
            args.embedding_tag,
            args.embedding_path_template,
        )
        target_embeddings = load_pretrained_embeddings(target_embedding_path).to(device)
        model.load_new_pretrain_embeddings(target_embeddings)

        loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False)
        recall_sum, ndcg_sum, total = evaluate_model_with_neg_sampling(
            model=model,
            dataloader=loader,
            top_k_set=topk,
            num_items=target_dataset.get_num_items(),
            device=device,
            num_negatives=args.eval_negatives,
            is_target_domain=True,
            is_sem_baseline=args.loss_mode == "sem",
            seed=args.seed,
            disable_tqdm=args.disable_tqdm,
        )

        metrics = metric_dict(recall_sum, ndcg_sum, total, topk)
        row = {
            "source_domain": args.source_domain,
            "target_domain": target,
            "model": args.model,
            "loss_mode": args.loss_mode,
            "seed": args.seed,
            **metrics,
        }
        rows.append(row)
        print(target, metrics)

    if rows:
        avg = {"source_domain": args.source_domain, "target_domain": "avg", "model": args.model, "loss_mode": args.loss_mode, "seed": args.seed}
        for key in rows[0]:
            if key.startswith(("R", "N")):
                avg[key] = float(np.mean([row[key] for row in rows]))
        rows.append(avg)

    append_results_csv(os.path.join(args.output_dir, "results.csv"), rows)


if __name__ == "__main__":
    main()
