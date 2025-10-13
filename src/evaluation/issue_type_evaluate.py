import warnings
warnings.filterwarnings("ignore")

import ast
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, hamming_loss
)
from sklearn.preprocessing import MultiLabelBinarizer


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


def to_list(x):
    if isinstance(x, list):
        return [str(i) for i in x]
    if pd.isna(x):
        return []
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                v = ast.literal_eval(s)
                if isinstance(v, (str, int)):
                    return [str(v)]
                return [str(i) for i in v]
            except Exception:
                return [s]
        if "," in s:
            return [t.strip() for t in s.split(",") if t.strip()]
        return [s]
    if isinstance(x, (int, float)):
        return [str(int(x))]
    try:
        return [str(i) for i in list(x)]
    except Exception:
        return []


def binarize_lists(true_lists, pred_lists):
    labels = sorted({lbl for row in true_lists for lbl in row} |
                    {lbl for row in pred_lists for lbl in row})
    mlb = MultiLabelBinarizer(classes=labels)
    y_true = mlb.fit_transform(true_lists)
    y_pred = mlb.transform(pred_lists)
    return y_true, y_pred, labels


df_gt = pd.read_csv(ROOT_PATH / "data" / true_df_path)
gt_df = df_gt[["file_name", "issue_type"]].copy()
gt_df.rename(columns={"issue_type": "issue_type_true"}, inplace=True)


def evaluate_model(input_setting: str, model_name: str, model_path: Path) -> dict | None:
    out_dir = ROOT_PATH / DATA_PATH / input_setting / model_path
    files = sorted(out_dir.glob("*.json"))
    if not files:
        print(f"[INFO] {model_name}: no json files at {out_dir}")
        return None

    rows = []
    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            rows.append({"file_name": p.stem, "issue_type_pred": data.get("issue_type", [])})
        except Exception as e:
            print(f"[WARN] parse fail: {p.name}: {e}")

    if not rows:
        print(f"[INFO] {model_name}: no valid rows")
        return None

    pred_df = pd.DataFrame(rows)

    merged = gt_df.merge(pred_df, on="file_name", how="inner")
    if merged.empty:
        print(f"[INFO] {model_name}: merged is empty")
        return None

    merged["issue_type_true"] = merged["issue_type_true"].apply(to_list)
    merged["issue_type_pred"] = merged["issue_type_pred"].apply(to_list)

    y_true, y_pred, labels = binarize_lists(
        merged["issue_type_true"].tolist(),
        merged["issue_type_pred"].tolist()
    )

    exact_match = accuracy_score(y_true, y_pred)
    precision_micro = precision_score(y_true, y_pred, average="micro", zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average="micro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)

    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    hamming = hamming_loss(y_true, y_pred)

    n_gt_total = len(gt_df)
    n_pred_files = len(files)
    n_merged = len(merged)
    coverage_vs_gt = n_merged / n_gt_total if n_gt_total else 0.0
    coverage_vs_pred = n_merged / n_pred_files if n_pred_files else 0.0

    return {
        "model": model_name,
        "n_gt_total": n_gt_total,
        "n_pred_files": n_pred_files,
        "n_eval_used": n_merged,
        "n_labels": len(labels),
        "coverage_vs_gt": round(coverage_vs_gt, 4),
        "coverage_vs_pred": round(coverage_vs_pred, 4),

        "exact_match": exact_match,

        "micro_precision": precision_micro,
        "micro_recall": recall_micro,
        "micro_f1": f1_micro,

        "macro_precision": precision_macro,
        "macro_recall": recall_macro,
        "macro_f1": f1_macro,

        "hamming_loss": hamming,
    }

input_settings = ["issue_type", "(merge)issue_type", "(claim)issue_type"]
for input_setting in input_settings:
    results = []
    for name, path in llm_dir.items():
        res = evaluate_model(input_setting, name, path)
        if res:
            results.append(res)

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        metric_cols = [
            "exact_match",
            "micro_precision", "micro_recall", "micro_f1",
            "macro_precision", "macro_recall", "macro_f1",
            "hamming_loss",
            "coverage_vs_gt", "coverage_vs_pred",
        ]
        results_df[metric_cols] = results_df[metric_cols].applymap(lambda x: round(float(x), 4))

        save_dir = ROOT_PATH / f"data/csv/evaluate_result/{input_setting}"
        save_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        save_path = save_dir / f"issue_type_eval_{ts}.csv"

        results_df.to_csv(save_path, index=False)
    else:
        print("[INFO] No results to save (no models produced evaluable outputs).")
