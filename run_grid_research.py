# run_auto.py
# Minimal hyperparameter grid search for SAGERec revision.
# Protocol:
#   - BERT4Rec backbone
#   - SAGERec with L_gen = L_SIC - beta * H_ID
#   - no sequence-level pattern evaluation
#   - fixed evaluation candidates
#   - seed = 2026 for search
#
# Output:
#   logs_grid_search/<timestamp>/<phase>/<run_tag>.log
#   logs_grid_search/<timestamp>/grid_search_summary.csv
#   results_grid_search/experiment_results.csv

import csv
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


# =========================
# User configuration
# =========================

PYTHON_EXE = "python"
MAIN_FILE = "sagerec_main.py"

SOURCE_DOMAIN = "amazon_movies_and_tv"
ALL_DOMAINS = "amazon_movies_and_tv,amazon_cds_and_vinyl,steam"
TARGET_DOMAINS = "amazon_cds_and_vinyl,steam"

MODEL_NAME = "bert4rec"
LOSS_MODE = "sage"
EMBEDDING_TAG = "llama"

SEED = 2026
EVAL_SEED = 2026

NUM_EPOCHS = 100
CHECK_STEP = 3
WARMUP_EPOCHS = 5
EARLY_STOP_PATIENCE = 10

BATCH_SIZE = 128
GAMMA_G = 0.1
TAU = 0.2

# Minimal sequential search grid.
BETA_GRID = [0.0, 0.01, 0.03, 0.05, 0.1]
THRESHOLD_GRID = [-0.05, 0.0, 0.05]
LAMBDA_GRID = [0.01, 0.03, 0.05, 0.08]

# Default values before each phase selects better ones.
DEFAULT_LAMBDA = 0.05
DEFAULT_BETA = 0.1
DEFAULT_THRESHOLD = 0.0

# Set to True for a one-job smoke test.
SMOKE_TEST = False

# If True, do not actually run commands, only print them.
DRY_RUN = False


# =========================
# Output directories
# =========================

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_ROOT = Path("logs_grid_search") / f"grid_{TIMESTAMP}"
RESULTS_DIR = Path("results_grid_search")
CKPT_DIR = Path("saved_ckpts_grid_search")
EVAL_CANDIDATE_DIR = Path("eval_candidates_grid_search")

SUMMARY_CSV = LOG_ROOT / "grid_search_summary.csv"
RESULT_CSV = RESULTS_DIR / "experiment_results.csv"


def safe_float_tag(x):
    s = f"{x:g}"
    return s.replace("-", "m").replace(".", "p")


def ensure_dirs():
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)


def write_summary_header():
    with open(SUMMARY_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "phase",
                "run_tag",
                "lambda_g",
                "gamma_g",
                "beta_id",
                "tau",
                "sim_threshold",
                "seed",
                "status",
                "elapsed_min",
                "R10",
                "N10",
                "R20",
                "N20",
                "score_R10_plus_N10",
                "log_file",
            ],
        )
        writer.writeheader()


def append_summary(row):
    with open(SUMMARY_CSV, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "phase",
                "run_tag",
                "lambda_g",
                "gamma_g",
                "beta_id",
                "tau",
                "sim_threshold",
                "seed",
                "status",
                "elapsed_min",
                "R10",
                "N10",
                "R20",
                "N20",
                "score_R10_plus_N10",
                "log_file",
            ],
        )
        writer.writerow(row)


def build_command(run_tag, lambda_g, beta_id, sim_threshold):
    return [
        PYTHON_EXE,
        "-u",
        MAIN_FILE,
        "--dataset_name", SOURCE_DOMAIN,
        "--all_domains", ALL_DOMAINS,
        "--target_domains", TARGET_DOMAINS,
        "--model_name", MODEL_NAME,
        "--loss_mode", LOSS_MODE,
        "--embedding_tag", EMBEDDING_TAG,
        "--batch_size", str(BATCH_SIZE),
        "--num_epochs", str(NUM_EPOCHS),
        "--check_step", str(CHECK_STEP),
        "--warmup_epochs", str(WARMUP_EPOCHS),
        "--early_stop_patience", str(EARLY_STOP_PATIENCE),
        "--early_stop_criterion", "zero_shot",
        "--lambda_g", str(lambda_g),
        "--gamma_g", str(GAMMA_G),
        "--beta_id", str(beta_id),
        "--tau", str(TAU),
        "--sim_threshold", str(sim_threshold),
        "--sage_id_sign", "minus",
        "--seed", str(SEED),
        "--fixed_eval_candidates",
        "--eval_candidate_seed", str(EVAL_SEED),
        "--eval_candidate_dir", str(EVAL_CANDIDATE_DIR),
        "--disable_tqdm",
        "--run_tag", run_tag,
        "--results_dir", str(RESULTS_DIR),
        "--model_path", str(CKPT_DIR),
        "--force_training",
    ]


def should_echo_to_console(line):
    keys = [
        "[CONFIG]",
        "Epoch ",
        "[Target]",
        "[FINAL REPORT]",
        "[RESULT]",
        "[GPU]",
        "Early Stop",
        "Recall@10",
        "NDCG@10",
    ]
    return any(k in line for k in keys)


def read_tail(path, n=80):
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except Exception as exc:
        return f"[Failed to read log tail: {exc}]"


def run_one(phase, lambda_g, beta_id, sim_threshold):
    phase_dir = LOG_ROOT / phase
    phase_dir.mkdir(parents=True, exist_ok=True)

    run_tag = (
        f"{phase}_"
        f"lam{safe_float_tag(lambda_g)}_"
        f"beta{safe_float_tag(beta_id)}_"
        f"th{safe_float_tag(sim_threshold)}_"
        f"seed{SEED}"
    )
    log_file = phase_dir / f"{run_tag}.log"
    cmd = build_command(run_tag, lambda_g, beta_id, sim_threshold)

    print("")
    print("------------------------------------------------------------")
    print(f"[START] {phase} | run_tag={run_tag}")
    print(f"lambda_g={lambda_g}, beta_id={beta_id}, threshold={sim_threshold}, seed={SEED}")
    print(f"log_file={log_file}")
    print("------------------------------------------------------------")

    if DRY_RUN:
        print(" ".join(cmd))
        return {
            "phase": phase,
            "run_tag": run_tag,
            "lambda_g": lambda_g,
            "beta_id": beta_id,
            "sim_threshold": sim_threshold,
            "status": "dry_run",
            "elapsed_min": 0.0,
            "R10": "",
            "N10": "",
            "R20": "",
            "N20": "",
            "score": -1.0,
            "log_file": str(log_file),
        }

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["TQDM_DISABLE"] = "1"

    start = time.time()

    with open(log_file, "w", encoding="utf-8-sig", errors="replace", newline="") as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )

        assert process.stdout is not None
        for line in process.stdout:
            f.write(line)
            f.flush()
            if should_echo_to_console(line):
                print("  " + line.rstrip())

        process.wait()

    elapsed_min = round((time.time() - start) / 60.0, 2)

    if process.returncode != 0:
        print(f"[FAILED] run_tag={run_tag}, exit_code={process.returncode}")
        print("Last 80 log lines:")
        print(read_tail(log_file, 80))

        result = {
            "phase": phase,
            "run_tag": run_tag,
            "lambda_g": lambda_g,
            "beta_id": beta_id,
            "sim_threshold": sim_threshold,
            "status": f"failed_{process.returncode}",
            "elapsed_min": elapsed_min,
            "R10": "",
            "N10": "",
            "R20": "",
            "N20": "",
            "score": -1.0,
            "log_file": str(log_file),
        }
        write_result_to_summary(result)
        raise RuntimeError(f"Job failed: {run_tag}")

    metrics = extract_avg_metrics(run_tag)
    score = metrics["R10"] + metrics["N10"]

    result = {
        "phase": phase,
        "run_tag": run_tag,
        "lambda_g": lambda_g,
        "beta_id": beta_id,
        "sim_threshold": sim_threshold,
        "status": "success",
        "elapsed_min": elapsed_min,
        "R10": metrics["R10"],
        "N10": metrics["N10"],
        "R20": metrics["R20"],
        "N20": metrics["N20"],
        "score": score,
        "log_file": str(log_file),
    }
    write_result_to_summary(result)

    print(
        f"[DONE] {run_tag} | "
        f"R10={metrics['R10']:.4f}, N10={metrics['N10']:.4f}, "
        f"R20={metrics['R20']:.4f}, N20={metrics['N20']:.4f}, "
        f"score={score:.4f}, elapsed={elapsed_min}min"
    )
    return result


def write_result_to_summary(result):
    append_summary({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phase": result["phase"],
        "run_tag": result["run_tag"],
        "lambda_g": result["lambda_g"],
        "gamma_g": GAMMA_G,
        "beta_id": result["beta_id"],
        "tau": TAU,
        "sim_threshold": result["sim_threshold"],
        "seed": SEED,
        "status": result["status"],
        "elapsed_min": result["elapsed_min"],
        "R10": result["R10"],
        "N10": result["N10"],
        "R20": result["R20"],
        "N20": result["N20"],
        "score_R10_plus_N10": result["score"],
        "log_file": result["log_file"],
    })


def extract_avg_metrics(run_tag):
    """
    Read results_grid_search/experiment_results.csv and return avg row metrics.
    """
    if not RESULT_CSV.exists():
        raise FileNotFoundError(f"Result CSV not found: {RESULT_CSV}")

    matched = []
    with open(RESULT_CSV, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_name = row.get("run_name", "")
            target_domain = row.get("target_domain", "")
            if run_tag in run_name and target_domain == "avg":
                matched.append(row)

    if not matched:
        raise RuntimeError(
            f"No avg row found for run_tag={run_tag} in {RESULT_CSV}. "
            "Please inspect the corresponding log file."
        )

    row = matched[-1]

    def get_float(key):
        value = row.get(key, "")
        if value is None or value == "":
            return 0.0
        return float(value)

    return {
        "R10": get_float("R10"),
        "N10": get_float("N10"),
        "R20": get_float("R20"),
        "N20": get_float("N20"),
    }


def choose_best(results, phase_name):
    successful = [r for r in results if r["status"] == "success"]
    if not successful:
        raise RuntimeError(f"No successful jobs in {phase_name}")

    best = max(successful, key=lambda x: (x["score"], x["R10"], x["N10"]))
    print("")
    print(f"[BEST] {phase_name}")
    print(
        f"run_tag={best['run_tag']} | "
        f"lambda_g={best['lambda_g']}, beta_id={best['beta_id']}, "
        f"threshold={best['sim_threshold']} | "
        f"R10={best['R10']:.4f}, N10={best['N10']:.4f}, "
        f"score={best['score']:.4f}"
    )
    return best


def main():
    ensure_dirs()
    write_summary_header()

    print("============================================================")
    print("Minimal SAGERec hyperparameter grid search")
    print(f"Log root: {LOG_ROOT}")
    print(f"Summary: {SUMMARY_CSV}")
    print(f"Result CSV: {RESULT_CSV}")
    print("Protocol: sage_id_sign=minus, no sequence-level pattern eval, fixed candidates")
    print("Selection score: avg R@10 + avg N@10")
    print("============================================================")

    if SMOKE_TEST:
        print("[SMOKE_TEST] Only one beta job will be executed.")
        beta_values = [DEFAULT_BETA]
        threshold_values = [DEFAULT_THRESHOLD]
        lambda_values = [DEFAULT_LAMBDA]
    else:
        beta_values = BETA_GRID
        threshold_values = THRESHOLD_GRID
        lambda_values = LAMBDA_GRID

    # Phase 1: beta search
    beta_results = []
    for beta in beta_values:
        beta_results.append(
            run_one(
                phase="phase1_beta",
                lambda_g=DEFAULT_LAMBDA,
                beta_id=beta,
                sim_threshold=DEFAULT_THRESHOLD,
            )
        )

    best_beta_job = choose_best(beta_results, "phase1_beta")
    best_beta = best_beta_job["beta_id"]

    # Phase 2: threshold search
    threshold_results = []
    for threshold in threshold_values:
        threshold_results.append(
            run_one(
                phase="phase2_threshold",
                lambda_g=DEFAULT_LAMBDA,
                beta_id=best_beta,
                sim_threshold=threshold,
            )
        )

    best_threshold_job = choose_best(threshold_results, "phase2_threshold")
    best_threshold = best_threshold_job["sim_threshold"]

    # Phase 3: lambda search
    lambda_results = []
    for lambda_g in lambda_values:
        lambda_results.append(
            run_one(
                phase="phase3_lambda",
                lambda_g=lambda_g,
                beta_id=best_beta,
                sim_threshold=best_threshold,
            )
        )

    best_lambda_job = choose_best(lambda_results, "phase3_lambda")

    print("")
    print("============================================================")
    print("Grid search finished.")
    print("Recommended final hyperparameters:")
    print(f"lambda_g      = {best_lambda_job['lambda_g']}")
    print(f"gamma_g       = {GAMMA_G}")
    print(f"beta_id       = {best_beta}")
    print(f"tau           = {TAU}")
    print(f"sim_threshold = {best_threshold}")
    print(f"seed          = {SEED}")
    print("")
    print(f"Full logs are in: {LOG_ROOT}")
    print(f"Summary CSV is: {SUMMARY_CSV}")
    print(f"Full result CSV is: {RESULT_CSV}")
    print("============================================================")


if __name__ == "__main__":
    main()
