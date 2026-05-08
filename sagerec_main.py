import argparse
import csv
import inspect
import json
import logging
import os
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from rec_datasets import AmazonUserSequencesDataset, SteamDataset, AuxiliarySemanticSampler
from utils import load_pretrained_embeddings, resolve_embedding_path
from models.model_trainer import evaluate_model_with_neg_sampling, train_sagerec_model
from models.structural_metrics import evaluate_full_catalog_structural

from baselines.sasrec import SASRecWithDomainAlignment
from baselines.bert4rec import BERT4RecWithDomainAlignment
from baselines.unisrec import UniSRecWithDomainAlignment
from baselines.gru4rec import GRU4RecWithDomainAlignment


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ======================================================================
# 【修改 1】彻底取消全局写死 ALL_DOMAINS。
# 后续所有域列表都从 --all_domains / --aux_domains / --target_domains 解析得到。
# 这样可以直接扩展到 AMT -> Grocery / Clothing 等远距离迁移实验。
# ======================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_comma_list(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    value = value.strip()
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def safe_float_str(x: float) -> str:
    """把浮点数转为适合文件名的字符串，例如 0.05 -> 0p05, -0.1 -> m0p1。"""
    s = f"{x:g}"
    return s.replace("-", "m").replace(".", "p")


def parse_args():
    parser = argparse.ArgumentParser()

    # -------------------------
    # 基础模型参数
    # -------------------------
    parser.add_argument("--hidden_units", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--model_name", type=str, default="bert4rec",
                        choices=["sasrec", "bert4rec", "gru4rec", "unisrec"])
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)

    # -------------------------
    # 【修改 2】统一数据域配置
    # -------------------------
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory containing each domain folder.")
    parser.add_argument("--dataset_name", type=str, default="amazon_movies_and_tv",
                        help="Source domain name.")
    parser.add_argument(
        "--all_domains",
        type=str,
        default="amazon_movies_and_tv,amazon_cds_and_vinyl,steam",
        help="Comma-separated full domain list. Must include --dataset_name."
    )
    parser.add_argument(
        "--aux_domains",
        type=str,
        default="",
        help=(
            "Comma-separated auxiliary domains used for semantic regularization. "
            "If empty, all non-source domains in --all_domains are used."
        )
    )
    parser.add_argument(
        "--target_domains",
        type=str,
        default="amazon_cds_and_vinyl,steam",
        help=(
            "Comma-separated zero-shot evaluation target domains. "
            "If empty, all auxiliary domains are evaluated."
        )
    )

    parser.add_argument(
        "--aux_sample_mode",
        type=str,
        default="domain_uniform",
        choices=["domain_uniform", "global_uniform"],
        help="Auxiliary sampling mode. Use domain_uniform for DAG analysis."
    )

    # -------------------------
    # 【修改 3】统一 embedding 配置
    # -------------------------
    parser.add_argument(
        "--embedding_tag",
        type=str,
        default="llama",
        help=(
            "Embedding file tag. For example, --embedding_tag llama expects "
            "<domain>_embedding_llama.parquet; --embedding_tag bert expects "
            "<domain>_embedding_bert.parquet."
        )
    )
    parser.add_argument(
        "--embedding_path_template",
        type=str,
        default="",
        help=(
            "Optional template for embedding path. You may use {data_root}, "
            "{domain}, and {tag}. Example: "
            "{data_root}/{domain}/{domain}_embedding_{tag}.parquet"
        )
    )

    # -------------------------
    # checkpoint / result 输出
    # -------------------------
    parser.add_argument("--model_path", type=str, default="./saved_ckpts/")
    parser.add_argument("--results_dir", type=str, default="./results/")
    parser.add_argument(
        "--run_tag",
        type=str,
        default="",
        help="Optional human-readable tag added into checkpoint/result names."
    )
    parser.add_argument(
        "--compact_ckpt_name",
        action="store_true",
        help="Use old short checkpoint names. Not recommended for parameter sweeps."
    )

    parser.add_argument(
        "--disable_tqdm",
        action="store_true",
        help="Disable tqdm progress bars. Recommended for batch experiments and log files."
    )

    # -------------------------
    # loss 模式
    # -------------------------
    parser.add_argument(
        "--loss_mode",
        type=str,
        default="sage",
        choices=["sem", "recg", "sage", "sage_no_sa", "sage_no_id", "sage_no_adagen"],
    )

    # -------------------------
    # 早停与评测控制
    # -------------------------
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--early_stop_criterion", type=str, default="zero_shot",
                        choices=["loss", "zero_shot"])
    parser.add_argument("--check_step", type=int, default=1,
                        help="Evaluate every N epochs.")
    parser.add_argument("--warmup_epochs", type=int, default=1,
                        help="Skip zero-shot eval for first N epochs.")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training and only run evaluation.")
    parser.add_argument("--eval_checkpoint", type=str, default="",
                        help="Path to a specific checkpoint to evaluate.")
    parser.add_argument("--force_training", action="store_true")

    # -------------------------
    # 训练 / 评估采样参数
    # -------------------------
    parser.add_argument("--n_exps", type=int, default=8,
                        help="Number of MoE experts for UniSRec.")
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--num_samples", type=int, default=4096,
                        help="Auxiliary semantic samples per epoch/batch sampler.")
    parser.add_argument("--num_sequential_patterns", type=int, default=20)
    parser.add_argument("--train_num_negatives", type=int, default=5)
    parser.add_argument("--eval_num_negatives", type=int, default=100)
    parser.add_argument("--eval_topk", type=str, default="10,20",
                        help="Comma-separated K values, e.g., 10,20.")

    # -------------------------
    # 【修改 4】SAGERec 关键实验参数全部命令行化
    # -------------------------
    parser.add_argument("--lambda_g", type=float, default=0.05,
                        help="Base regularization strength.")
    parser.add_argument("--gamma_g", type=float, default=0.1,
                        help="Decay rate for domain discrepancy.")
    parser.add_argument("--beta_id", type=float, default=0.1,
                        help="Weight for intra-domain diversity.")
    parser.add_argument("--tau", type=float, default=0.2,
                        help="Temperature for diversity entropy.")
    parser.add_argument(
        "--sim_threshold",
        type=float,
        default=0.0,
        help=(
            "Similarity threshold for SIC. Requires models/loss.py to read "
            "args.sim_threshold or accept sim_threshold in unified_alignment_loss."
        )
    )

    parser.add_argument(
        "--sage_id_sign",
        type=str,
        default="minus",
        choices=["minus", "plus"],
        help=(
            "Audit option for SAGERec. "
            "minus: L_gen = L_SIC - beta*H_ID, consistent with diversity maximization. "
            "plus: L_gen = L_SIC + beta*H_ID, used only to audit previous results."
        )
    )

    parser.add_argument(
        "--use_seq_pattern_eval",
        action="store_true",
        help=(
            "Enable sequence-level pattern aggregation during target-domain evaluation. "
            "This is required for full LLM-RecG reproduction."
        )
    )

    # -------------------------
    # 【修改 5】预留固定候选评测参数
    # 说明：当前主函数会自动检测 evaluate_model_with_neg_sampling 是否支持这些参数。
    # 如果你尚未修改 model_trainer.py，它们会被安全忽略，不会报错。
    # -------------------------
    parser.add_argument("--fixed_eval_candidates", action="store_true",
                        help="Use fixed negative candidates if trainer supports it.")
    parser.add_argument("--eval_candidate_seed", type=int, default=2026)
    parser.add_argument("--eval_candidate_dir", type=str, default="./eval_candidates/")

    # -------------------------
    # 【修改 6】预留结构性指标开关
    # 说明：如果后续新增 structural_metrics.py，则可直接打开。
    # -------------------------
    parser.add_argument("--eval_structural", action="store_true",
                        help="Evaluate Coverage/ILD if structural_metrics.py exists.")
    parser.add_argument("--structural_topk", type=int, default=10)

    return parser.parse_args()


def build_domain_config(args) -> Dict[str, List[str]]:
    all_domains = parse_comma_list(args.all_domains)
    if not all_domains:
        raise ValueError("--all_domains cannot be empty.")
    if args.dataset_name not in all_domains:
        raise ValueError(
            f"Source domain {args.dataset_name} is not in --all_domains: {all_domains}"
        )

    aux_domains = parse_comma_list(args.aux_domains)
    if not aux_domains:
        aux_domains = [d for d in all_domains if d != args.dataset_name]

    target_domains = parse_comma_list(args.target_domains)
    if not target_domains:
        target_domains = list(aux_domains)

    for d in aux_domains + target_domains:
        if d not in all_domains:
            raise ValueError(f"Domain {d} must be included in --all_domains.")

    if args.dataset_name in aux_domains:
        raise ValueError("--aux_domains must not include the source domain.")
    if args.dataset_name in target_domains:
        raise ValueError("--target_domains must not include the source domain.")

    return {
        "all_domains": all_domains,
        "aux_domains": aux_domains,
        "target_domains": target_domains,
    }


def resolve_embedding_path_by_tag(dataset_name: str, args) -> str:
    """
    【修改 7】embedding 路径统一入口。
    优先使用 --embedding_path_template；否则按 tag 自动查找。
    如果 tag=llama 且使用默认 data_root，则兼容原 utils.resolve_embedding_path。
    """
    tag = args.embedding_tag.strip()

    if args.embedding_path_template:
        path = args.embedding_path_template.format(
            data_root=args.data_root,
            domain=dataset_name,
            tag=tag,
        )
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"Embedding file not found from template: {path}")

    candidates = [
        os.path.join(args.data_root, dataset_name, f"{dataset_name}_embedding_{tag}.parquet"),
    ]

    # 兼容旧命名：llama3 / llama
    if tag == "llama":
        candidates.extend([
            os.path.join(args.data_root, dataset_name, f"{dataset_name}_embedding_llama3.parquet"),
            os.path.join(args.data_root, dataset_name, f"{dataset_name}_embedding_llama.parquet"),
        ])

    for path in candidates:
        if os.path.exists(path):
            return path

    # 兼容原始 utils.resolve_embedding_path，仅在默认 data_root 下兜底
    if tag == "llama" and (args.data_root == "./data" or args.data_root == "data"):
        try:
            return resolve_embedding_path(dataset_name)
        except Exception:
            pass

    raise FileNotFoundError(
        "Cannot find embedding file. Tried:\n  " + "\n  ".join(candidates)
    )


def load_embeddings(dataset_name: str, args) -> torch.Tensor:
    path = resolve_embedding_path_by_tag(dataset_name, args)
    logger.info(f"[EMBEDDING] Loading {dataset_name} embeddings from: {path}")
    return load_pretrained_embeddings(path)


def initialize_dataset(dataset_name: str, max_len: int, data_root: str):
    data_path = os.path.join(data_root, dataset_name, "processed_data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data not found: {data_path}")

    data = pd.read_csv(data_path)
    if "amazon" in dataset_name.lower():
        return AmazonUserSequencesDataset(
            data=data,
            max_seq_length=max_len,
            dataset_name=dataset_name,
        )
    return SteamDataset(
        data=data[["UserId", "ItemId", "Timestamp"]],
        max_seq_length=max_len,
        dataset_name=dataset_name,
    )


def collect_source_train_sequences(dataset, device):
    train_sequences = [dataset[idx][0].unsqueeze(0) for idx in range(len(dataset))]
    return torch.cat(train_sequences, dim=0).to(device)


def maybe_extract_source_patterns(
        model,
        source_train_sequences: torch.Tensor,
        loss_mode: str,
        use_seq_pattern_eval: bool = False,
) -> None:
    """
    只有显式启用 sequence-level pattern evaluation 时才提取 source patterns。
    当前主实验和超参数搜索均不使用 SG，因此跳过该步骤可以显著节省时间，
    且不会改变 no-SG 结果。
    """
    if (
        use_seq_pattern_eval
        and loss_mode != "sem"
        and hasattr(model, "extract_sequential_patterns_from_source")
    ):
        model.extract_sequential_patterns_from_source(source_train_sequences)



def summarize_eval_result(result_dict: Dict[str, Dict[str, float]]) -> str:
    ordered_keys = [k for k in result_dict.keys() if k != "avg"]
    if "avg" in result_dict:
        ordered_keys += ["avg"]

    parts = []
    for key in ordered_keys:
        metrics = result_dict[key]
        metric_text = ", ".join(
            [f"{m}={metrics[m]:.4f}" for m in sorted(metrics.keys()) if m.startswith(("R", "N"))]
        )
        parts.append(f"{key}: {metric_text}")
    return " | ".join(parts)


def evaluate_with_optional_args(
    args,
    model,
    dataloader,
    top_k_set: List[int],
    num_items: int,
    device,
    is_target_domain: bool,
    is_sem_baseline: bool,
    dataset_name: str = "",
):
    """
    【修改 8】评估函数兼容层。
    如果你之后在 model_trainer.py 中增加 fixed candidates 支持，这里会自动传参；
    如果还没改，这里会自动忽略，保证主函数可运行。
    """
    kwargs = dict(
        model=model,
        dataloader=dataloader,
        top_k_set=top_k_set,
        num_items=num_items,
        device=device,
        num_negatives=args.eval_num_negatives,
        is_target_domain=is_target_domain,
        is_sem_baseline=is_sem_baseline,
    )

    sig = inspect.signature(evaluate_model_with_neg_sampling)
    if "fixed_eval_candidates" in sig.parameters:
        kwargs["fixed_eval_candidates"] = args.fixed_eval_candidates
    if "eval_candidate_seed" in sig.parameters:
        kwargs["eval_candidate_seed"] = args.eval_candidate_seed
    if "eval_candidate_dir" in sig.parameters:
        kwargs["eval_candidate_dir"] = args.eval_candidate_dir
    if "dataset_name" in sig.parameters:
        kwargs["dataset_name"] = dataset_name
    if "disable_tqdm" in sig.parameters:
        kwargs["disable_tqdm"] = getattr(args, "disable_tqdm", False)
    if "use_seq_pattern_eval" in sig.parameters:
        kwargs["use_seq_pattern_eval"] = getattr(args, "use_seq_pattern_eval", False)

    return evaluate_model_with_neg_sampling(**kwargs)


def compute_metric_dict(recall_sum, ndcg_sum, total: int, top_k_set: List[int]) -> Dict[str, float]:
    metrics = {}
    for k in top_k_set:
        metrics[f"R{k}"] = (recall_sum[k] / max(total, 1)) * 100.0
        metrics[f"N{k}"] = (ndcg_sum[k] / max(total, 1)) * 100.0
    return metrics


def build_zero_shot_eval_fn(
    args,
    device,
    source_embeddings: torch.Tensor,
    source_train_sequences: torch.Tensor,
    top_k_set: List[int],
):
    def zero_shot_eval_fn(model, target_domains: List[str]) -> Dict[str, Dict[str, float]]:
        model.eval()
        maybe_extract_source_patterns(
            model,
            source_train_sequences,
            args.loss_mode,
            getattr(args, "use_seq_pattern_eval", False),
        )

        results = {}
        with torch.no_grad():
            for target_name in target_domains:
                target_dataset = initialize_dataset(target_name, args.max_len, args.data_root)
                target_embs = load_embeddings(target_name, args)
                model.load_new_pretrain_embeddings(target_embs)
                target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False)

                recall_sum, ndcg_sum, total = evaluate_with_optional_args(
                    args=args,
                    model=model,
                    dataloader=target_loader,
                    top_k_set=top_k_set,
                    num_items=target_dataset.get_num_items(),
                    device=device,
                    is_target_domain=True,
                    is_sem_baseline=args.loss_mode == "sem",
                    dataset_name=target_name,
                )
                results[target_name] = compute_metric_dict(recall_sum, ndcg_sum, total, top_k_set)

        model.load_new_pretrain_embeddings(source_embeddings)

        # 【修复】先固定 target 结果列表，再计算 avg。
        # 避免 results["avg"] 被加入 results.values() 后又参与自身均值计算。
        if results:
            target_metric_dicts = list(results.values())

            avg_metrics = {}
            for k in top_k_set:
                avg_metrics[f"R{k}"] = float(
                    np.mean([v[f"R{k}"] for v in target_metric_dicts if f"R{k}" in v])
                )
                avg_metrics[f"N{k}"] = float(
                    np.mean([v[f"N{k}"] for v in target_metric_dicts if f"N{k}" in v])
                )

            results["avg"] = avg_metrics
        else:
            results["avg"] = {}
            for k in top_k_set:
                results["avg"][f"R{k}"] = 0.0
                results["avg"][f"N{k}"] = 0.0

        return results

    return zero_shot_eval_fn


def build_model(args, source_embeddings: torch.Tensor, device):
    model_name_lower = args.model_name.lower()

    if model_name_lower == "sasrec":
        return SASRecWithDomainAlignment(
            hidden_units=args.hidden_units,
            max_seq_length=args.max_len,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout_rate=args.dropout_rate,
            pretrained_item_embeddings=source_embeddings,
            num_sequential_patterns=args.num_sequential_patterns,
        ).to(device)

    if model_name_lower == "bert4rec":
        return BERT4RecWithDomainAlignment(
            hidden_units=args.hidden_units,
            max_seq_length=args.max_len,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout_rate=args.dropout_rate,
            pretrained_item_embeddings=source_embeddings,
            num_sequential_patterns=args.num_sequential_patterns,
        ).to(device)

    if model_name_lower == "gru4rec":
        return GRU4RecWithDomainAlignment(
            hidden_units=args.hidden_units,
            num_layers=args.num_layers,
            dropout_rate=args.dropout_rate,
            pretrained_item_embeddings=source_embeddings,
            num_sequential_patterns=args.num_sequential_patterns,
            max_seq_length=args.max_len,
        ).to(device)

    if model_name_lower == "unisrec":
        return UniSRecWithDomainAlignment(
            hidden_units=args.hidden_units,
            max_seq_length=args.max_len,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout_rate=args.dropout_rate,
            pretrained_item_embeddings=source_embeddings,
            num_sequential_patterns=args.num_sequential_patterns,
            n_exps=args.n_exps,
        ).to(device)

    raise ValueError(f"Unsupported model name: {args.model_name}")


def build_run_name(args) -> str:
    if args.compact_ckpt_name:
        return f"{args.model_name.lower()}_{args.dataset_name}_{args.loss_mode}"

    parts = [
        args.model_name.lower(),
        f"src_{args.dataset_name}",
        args.loss_mode,
        f"emb_{args.embedding_tag}",
        f"bs{args.batch_size}",
        f"lam{safe_float_str(args.lambda_g)}",
        f"gam{safe_float_str(args.gamma_g)}",
        f"beta{safe_float_str(args.beta_id)}",
        f"tau{safe_float_str(args.tau)}",
        f"th{safe_float_str(args.sim_threshold)}",
        f"seed{args.seed}",
    ]
    if args.run_tag:
        parts.append(args.run_tag)
    return "_".join(parts)


def save_config(args, domain_config: Dict[str, List[str]], run_name: str) -> str:
    os.makedirs(args.results_dir, exist_ok=True)
    config_path = os.path.join(args.results_dir, f"{run_name}_config.json")
    payload = vars(args).copy()
    payload["resolved_all_domains"] = domain_config["all_domains"]
    payload["resolved_aux_domains"] = domain_config["aux_domains"]
    payload["resolved_target_domains"] = domain_config["target_domains"]
    payload["created_at"] = datetime.now().isoformat(timespec="seconds")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return config_path


def get_peak_gpu_mem_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / 1024 / 1024)


def append_results_csv(
    args,
    run_name: str,
    final_results: Dict[str, Dict[str, float]],
    csv_path: str,
    peak_gpu_mem_mb: float,
) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    base_fields = [
        "timestamp", "run_name", "source_domain", "target_domain",
        "all_domains", "aux_domains", "model_name", "loss_mode",
        "embedding_tag", "batch_size", "lambda_g", "gamma_g", "beta_id",
        "tau", "sim_threshold", "seed", "eval_num_negatives",
        "fixed_eval_candidates", "eval_candidate_seed", "peak_gpu_mem_mb",
        "sage_id_sign", "use_seq_pattern_eval",
    ]

    metric_fields = sorted({
        metric for target, metrics in final_results.items()
        for metric in metrics.keys()
        if metric.startswith(("R", "N", "Coverage", "ILD"))
    })

    fieldnames = base_fields + metric_fields
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for target_name, metrics in final_results.items():
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "run_name": run_name,
                "source_domain": args.dataset_name,
                "target_domain": target_name,
                "all_domains": args.all_domains,
                "aux_domains": args.aux_domains,
                "model_name": args.model_name,
                "loss_mode": args.loss_mode,
                "embedding_tag": args.embedding_tag,
                "batch_size": args.batch_size,
                "lambda_g": args.lambda_g,
                "gamma_g": args.gamma_g,
                "beta_id": args.beta_id,
                "tau": args.tau,
                "sim_threshold": args.sim_threshold,
                "seed": args.seed,
                "eval_num_negatives": args.eval_num_negatives,
                "fixed_eval_candidates": int(args.fixed_eval_candidates),
                "eval_candidate_seed": args.eval_candidate_seed,
                "peak_gpu_mem_mb": peak_gpu_mem_mb,
                "sage_id_sign": getattr(args, "sage_id_sign", "minus"),
                "use_seq_pattern_eval": int(getattr(args, "use_seq_pattern_eval", False)),
            }
            row.update(metrics)
            writer.writerow(row)


def try_structural_eval(args, model, target_loader, target_dataset, device, target_name: str) -> Dict[str, float]:
    """
    【修改 9】预留结构性指标接口。
    后续你只要新增 structural_metrics.py 并实现 evaluate_full_catalog_structural，
    主函数无需再改。
    """
    if not args.eval_structural:
        return {}

    # try:
    #     from structural_metrics import evaluate_full_catalog_structural
    # except Exception as exc:
    #     logger.warning(
    #         f"[STRUCTURAL] structural_metrics.py not available. Skip Coverage/ILD. Error: {exc}"
    #     )
    #     return {}

    try:
        coverage, ild = evaluate_full_catalog_structural(
            model=model,
            dataloader=target_loader,
            num_items=target_dataset.get_num_items(),
            device=device,
            top_k=args.structural_topk,
            is_sem_baseline=args.loss_mode == "sem",
        )
        return {
            f"Coverage{args.structural_topk}": float(coverage) * 100.0,
            f"ILD{args.structural_topk}": float(ild),
        }
    except Exception as exc:
        logger.warning(f"[STRUCTURAL] Failed on {target_name}: {exc}")
        return {}


def main():
    args = parse_args()
    domain_config = build_domain_config(args)
    top_k_set = [int(k) for k in parse_comma_list(args.eval_topk)]

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.eval_candidate_dir, exist_ok=True)

    run_name = build_run_name(args)
    config_path = save_config(args, domain_config, run_name)

    logger.info("=" * 80)
    logger.info(f"[CONFIG] run_name = {run_name}")
    logger.info(f"[CONFIG] config saved to: {config_path}")
    logger.info(f"[CONFIG] source = {args.dataset_name}")
    logger.info(f"[CONFIG] all_domains = {domain_config['all_domains']}")
    logger.info(f"[CONFIG] aux_domains = {domain_config['aux_domains']}")
    logger.info(f"[CONFIG] target_domains = {domain_config['target_domains']}")
    logger.info(f"[CONFIG] embedding_tag = {args.embedding_tag}")
    logger.info(f"[CONFIG] sim_threshold = {args.sim_threshold}")
    logger.info("=" * 80)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    source_dataset = initialize_dataset(args.dataset_name, args.max_len, args.data_root)
    source_embeddings = load_embeddings(args.dataset_name, args)
    num_items = source_dataset.get_num_items()

    train_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(args, source_embeddings, device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    model_save_path = os.path.join(args.model_path, f"{run_name}.pth")

    all_domains = domain_config["all_domains"]
    aux_names = domain_config["aux_domains"]
    target_names = domain_config["target_domains"]
    current_domain_id = all_domains.index(args.dataset_name)

    # logger.info("Initializing AuxiliarySemanticSampler for dynamic global sampling...")

    # 【修改 10】Sampler 参数兼容层：
    # 如果你后续在 rec_datasets.AuxiliarySemanticSampler 中新增 embedding_tag /
    # embedding_path_template，这里会自动传入；如果还没改，也不会报错。
    sampler_kwargs = dict(
        model=model,
        aux_domains=aux_names,
        all_domains_for_index=all_domains,
        data_root=args.data_root,
        aux_batch_size=args.num_samples,
        sample_mode=args.aux_sample_mode,
        verbose=True,
    )
    sampler_sig = inspect.signature(AuxiliarySemanticSampler)
    if "embedding_tag" in sampler_sig.parameters:
        sampler_kwargs["embedding_tag"] = args.embedding_tag
    if "embedding_path_template" in sampler_sig.parameters:
        sampler_kwargs["embedding_path_template"] = args.embedding_path_template

    if args.loss_mode == "sem":
        logger.info("Skip AuxiliarySemanticSampler because loss_mode=sem.")
        aux_sampler = None
    else:
        logger.info("Initializing AuxiliarySemanticSampler for dynamic global sampling...")
        aux_sampler = AuxiliarySemanticSampler(**sampler_kwargs)

    source_train_sequences = collect_source_train_sequences(source_dataset, device)

    saved_paths = {"loss": model_save_path.replace(".pth", "_loss.pth")}
    for target in target_names:
        saved_paths[f"test_on_{target}"] = model_save_path.replace(".pth", f"_test_on_{target}.pth")

    if args.eval_only:
        logger.info("Skipping training because --eval_only is set.")
    elif os.path.exists(saved_paths["loss"]) and not args.force_training:
        logger.info("Found existing checkpoints. Skipping training.")
    else:
        zero_shot_eval_fn = build_zero_shot_eval_fn(
            args=args,
            device=device,
            source_embeddings=source_embeddings,
            source_train_sequences=source_train_sequences,
            top_k_set=top_k_set,
        )

        saved_paths = train_sagerec_model(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            num_epochs=args.num_epochs,
            num_items=num_items,
            num_aux_domains=len(all_domains),
            current_domain_id=current_domain_id,
            aux_sampler=aux_sampler,
            alpha_base=args.alpha,
            early_stop_patience=args.early_stop_patience,
            model_save_path=model_save_path,
            device=device,
            train_num_negatives=args.train_num_negatives,
            temperature=0.1,
            target_domains=target_names,
            eval_fn=zero_shot_eval_fn,
            check_step=args.check_step,
            warmup_epochs=args.warmup_epochs,
            early_stop_criterion=args.early_stop_criterion,
            loss_mode=args.loss_mode,
            args=args,  # 【关键】loss.py / trainer.py 后续读取 sim_threshold 等参数都从这里取。
        )

    # ------------------------------------------------------------------
    # 独立评估：In-domain
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("[EVAL] IN-DOMAIN (Based on best Training Loss)")
    logger.info("=" * 60)

    loss_ckpt = args.eval_checkpoint if args.eval_checkpoint else saved_paths["loss"]
    if os.path.exists(loss_ckpt):
        model.load_state_dict(torch.load(loss_ckpt, map_location=device))
        evaluate_with_optional_args(
            args=args,
            model=model,
            dataloader=eval_loader,
            top_k_set=top_k_set,
            num_items=num_items,
            device=device,
            is_target_domain=False,
            is_sem_baseline=args.loss_mode == "sem",
            dataset_name=args.dataset_name,
        )
    else:
        logger.warning(f"Checkpoint not found for In-Domain eval: {loss_ckpt}")

    # ------------------------------------------------------------------
    # 独立评估：Zero-shot transfer
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("[EVAL] ZERO-SHOT TRANSFER")
    logger.info("=" * 60)

    final_results: Dict[str, Dict[str, float]] = {}

    for target_name in target_names:
        model.load_new_pretrain_embeddings(source_embeddings)

        if args.early_stop_criterion == "loss":
            target_ckpt = args.eval_checkpoint if args.eval_checkpoint else saved_paths["loss"]
            logger.info(f"\n>>> Testing on: {target_name} (Using best LOSS model)")
        else:
            target_ckpt = args.eval_checkpoint if args.eval_checkpoint else saved_paths.get(f"test_on_{target_name}")
            logger.info(f"\n>>> Strictly testing on: {target_name} (Model selected via other domains)")

        if os.path.exists(target_ckpt):
            model.load_state_dict(torch.load(target_ckpt, map_location=device))
        else:
            logger.warning(f"Checkpoint not found for Zero-Shot eval: {target_ckpt}")
            continue

        maybe_extract_source_patterns(
            model,
            source_train_sequences,
            args.loss_mode,
            getattr(args, "use_seq_pattern_eval", False),
        )

        target_dataset = initialize_dataset(target_name, args.max_len, args.data_root)
        target_embs = load_embeddings(target_name, args)
        model.load_new_pretrain_embeddings(target_embs)

        target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False)
        recall_sum, ndcg_sum, total = evaluate_with_optional_args(
            args=args,
            model=model,
            dataloader=target_loader,
            top_k_set=top_k_set,
            num_items=target_dataset.get_num_items(),
            device=device,
            is_target_domain=True,
            is_sem_baseline=args.loss_mode == "sem",
            dataset_name=target_name,
        )

        metrics = compute_metric_dict(recall_sum, ndcg_sum, total, top_k_set)
        metrics.update(try_structural_eval(args, model, target_loader, target_dataset, device, target_name))
        final_results[target_name] = metrics

    if final_results:
        final_results["avg"] = {}
        metric_names = sorted({m for metrics in final_results.values() for m in metrics.keys()})
        for metric_name in metric_names:
            final_results["avg"][metric_name] = float(
                np.mean([v[metric_name] for v in final_results.values() if metric_name in v])
            )

    peak_gpu_mem_mb = get_peak_gpu_mem_mb()
    logger.info(f"[GPU] peak_gpu_mem_mb = {peak_gpu_mem_mb:.2f}")
    logger.info(f"[FINAL REPORT] {summarize_eval_result(final_results)}")

    result_csv = os.path.join(args.results_dir, "experiment_results.csv")
    append_results_csv(
        args=args,
        run_name=run_name,
        final_results=final_results,
        csv_path=result_csv,
        peak_gpu_mem_mb=peak_gpu_mem_mb,
    )
    logger.info(f"[RESULT] appended to: {result_csv}")


if __name__ == "__main__":
    main()
