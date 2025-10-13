import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, accuracy_score,
    precision_score, recall_score
)

ROOT_PATH = Path("Path/root/dir")
DATA_PATH = Path("data/json/decision_predict/output")
true_df_path = Path("csv/20250818_ptab.csv")

llm_dir = {
    "claude" : Path("claude-sonnet-4-20250514"),
    "gemini_25_pro": Path("gemini-2.5-pro"),
    "gemini_15_pro": Path("gemini-1.5-pro"),
    "gpt_4o": Path("gpt-4o-2024-08-06"),
    "gpt_o3": Path("o3-2025-04-16"),
    "solar": Path("solar-pro2"),
    "qwen": Path("Qwen/Qwen3-8B"),
    "llama": Path("meta-llama/Llama-3.1-8B-Instruct"),
    "mistral": Path("mistralai/Mistral-7B-Instruct-v0.3"),
    "deepseek": Path("deepseek-ai/deepseek-llm-7b-chat"),
    "t5": Path("google/t5gemma-2b-2b-ul2-it"),
}

subdecision_map_dir = Path("data/json/decision_type.json")

df = pd.read_csv(ROOT_PATH / "data" / true_df_path)
fine_sd_df = df[["file_name", "subdecision"]].copy()

fine_labels = json.loads((ROOT_PATH / subdecision_map_dir).read_text(encoding="utf-8")).keys()
fine_l2i = {k: i for i, k in enumerate(fine_labels)}

fine_sd_df["subdecision_true"] = fine_sd_df["subdecision"].map(fine_l2i)

def evaluate_model(input_setting: str, model_name: str, model_path: Path) -> dict | None:
    output_path = ROOT_PATH / DATA_PATH / input_setting / model_path
    files = sorted(output_path.glob("*.json"))
    if not files:
        print(f"[INFO] {model_name}: no json files at {output_path}")
        return None

    rows = []
    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] parse fail: {p.name}: {e}")
            continue
        row = {"file_name": p.stem}
        row.update(data)
        rows.append(row)

    if not rows:
        print(f"[INFO] {model_name}: no valid rows")
        return None

    pred_df = pd.DataFrame(rows)

    merged = fine_sd_df.merge(pred_df, on="file_name", how="inner")

    merged = merged[pd.to_numeric(merged["decision_number"], errors="coerce").notna()].copy()
    if merged.empty:
        print(f"[INFO] {model_name}: merged is empty after filtering")
        return None

    y_true = merged["subdecision_true"].astype(int).to_numpy()
    y_pred = merged["decision_number"].astype(int).to_numpy()

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)

    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    n_gt_total = len(fine_sd_df)
    n_pred_files = len(files)
    n_merged = len(merged)
    coverage_vs_gt = n_merged / n_gt_total if n_gt_total   else 0.0
    coverage_vs_pred = n_merged / n_pred_files if n_pred_files else 0.0

    return {
        "model": model_name,
        "n_gt_total": n_gt_total,
        "n_pred_files": n_pred_files,
        "n_eval_used": n_merged,
        "coverage_vs_gt": round(coverage_vs_gt, 4),
        "coverage_vs_pred": round(coverage_vs_pred, 4),

        "accuracy": acc,
        "balanced_acc": bacc,

        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,

        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
    }

input_settings = ["subdecision_type", "(merge)subdecision_type", "(claim)subdecision_type"]
for input_setting in input_settings:
    results = []
    for name, path in llm_dir.items():
        res = evaluate_model(input_setting, name, path)
        if res:
            results.append(res)

    results_df = pd.DataFrame(results)
    metric_cols = [
        "accuracy", "balanced_acc",
        "macro_precision", "macro_recall", "macro_f1",
        "micro_f1", "weighted_f1",
        "coverage_vs_gt", "coverage_vs_pred",
    ]
    results_df[metric_cols] = results_df[metric_cols].applymap(lambda x: round(float(x), 4))

    save_dir = ROOT_PATH / f"data/csv/evaluate_result/{input_setting}"
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = save_dir / f"subdecision_eval_{ts}.csv"

    results_df.to_csv(save_path, index=False)
    print(f"[SAVED] {save_path}")