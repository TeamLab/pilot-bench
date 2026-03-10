# 🧠 PILOT-Bench Classification  
*A Benchmark for Legal Reasoning in the Patent Domain with IRAC-Aligned Classification Tasks*

This repository provides code, configuration, and evaluation scripts for the **PILOT-Bench** paper:

> **Yehoon Jang\*, Chaewon Lee\*, Hyun-seok Min, and Sungchul Choi (2025)**  
> *PILOT-Bench: A Benchmark for Legal Reasoning in the Patent Domain with IRAC-Aligned Classification Tasks*  
> [Paper](https://aclanthology.org/2025.nllp-1.17/) | [Dataset](https://huggingface.co/datasets/Yehoon/pilot-bench)

---

## 🧩 Overview
PILOT-Bench evaluates the **legal reasoning capability of large language models (LLMs)** within the **U.S. Patent Trial and Appeal Board (PTAB)** domain.  
This repository focuses on **three IRAC-aligned classification tasks**:

| Task | IRAC Stage | Label Type | # Labels | Metric Type |
|------|-------------|-------------|-----------|--------------|
| **Issue Type** | Issue | Multi-label | 5 | Exact Match / Macro-F1 / Micro-F1 |
| **Board Authorities** | Rule | Multi-label | 10 | Exact Match / Macro-F1 / Micro-F1 |
| **Subdecision** | Conclusion | Multi-class | 23 (fine) / 6 (coarse) | Accuracy / Macro-F1 / Weighted-F1 |

All experiments follow a **zero-shot evaluation protocol**, using standardized prompts and unified input settings:
- **Split (Base):** Appellant vs. Examiner roles separated  
- **Merge:** Role-neutral concatenation  
- **Split + Claim:** Role-split inputs with appended claim text  

---

## 📁 Directory Structure
```
pilot-bench/
│
├── config/                # JSON configuration files for prediction and evaluation
│
├── data/                  # Input / output data (PTAB opinion-split JSONs)
│
├── src/
│   ├── evaluation/        # Task-specific evaluation scripts
│   │   ├── board_ruling_evaluate.py
│   │   ├── issue_type_evaluate.py
│   │   ├── subdecision_evaluate.py
│   │   ├── subdecision_coarse_evaluate.py
│   │   └── evaluate.sh
│   │
│   ├── inference/         # Inference scripts for each classification task
│   │   ├── board_ruling_predict.py
│   │   ├── decision_predict.py
│   │   └── issue_predict.py
│   │
│   ├── llms/              # LLM client wrappers (OpenAI, Gemini, Claude, etc.)
│   │
│   └── utils/             # Utility modules
│
└── README.md              # Project documentation
```

---

## ⚙️ Installation
```bash
git clone https://github.com/TeamLab/pilot-bench.git
cd pilot-bench
pip install -r requirements.txt
```

Python ≥ 3.10 is recommended.

---

## 🚀 Usage

### 🔹 1. Zero-shot / Prompt-based Inference 
#### 1.1. Async
```bash
python /Path/root/dir/src/inference/board_ruling_predict.py --config "/Path/root/dir/config/board_ruling_predict.json" --prompt "board_ruling" --wandb_entity "pilot-bench" --wandb_project "board_authorities_predict" --input_setting "base" --model "qwen" --mode "async"
```

#### 1.2. Batch
```bash
python /Path/root/dir/src/inference/board_ruling_predict.py --config "/Path/root/dir/config/board_ruling_predict.json" --prompt "board_ruling" --wandb_entity "pilot-bench" --wandb_project "board_authorities_predict" --input_setting "base" --model "gpt" --mode "batch"
```

### 🔹 2. Evalaute
```bash
python Path/root/dir/src/evaluation/subdecision_evaluate.py
```

### 🔹 3. Evaluation Metrics

All metrics are computed using `sklearn.metrics` (with `zero_division=0` to handle undefined cases).  
Each evaluation script also reports **coverage statistics** to indicate the ratio of evaluated samples relative to the ground truth and predictions.

#### ✅ Multi-label tasks (Issue Type / Board Authorities)

- `exact_match` — **Subset accuracy**, i.e., the proportion of samples whose predicted label set exactly matches the ground truth (`accuracy_score` on binarized matrices).  
- `micro_precision`, `micro_recall`, `micro_f1` — **Micro-averaged** metrics aggregating TP/FP/FN across all labels (`average="micro"`).  
- `macro_precision`, `macro_recall`, `macro_f1` — **Macro-averaged** metrics computed as the unweighted mean across all labels (`average="macro"`).  
- `hamming_loss` — The fraction of incorrect labels among all possible label assignments.  
- `coverage_vs_gt` — Ratio of evaluated samples to total ground-truth samples (`n_eval_used / n_gt_total`).  
- `coverage_vs_pred` — Ratio of evaluated samples to total prediction files (`n_eval_used / n_pred_files`).  

Additional diagnostic statistics:  
- `n_gt_total` — Number of ground-truth entries.  
- `n_pred_files` — Number of prediction JSON files.  
- `n_eval_used` — Number of samples successfully merged for evaluation.  
- `n_labels` — Number of unique labels in the task.

#### ✅ Multi-class tasks (Subdecision)

- `accuracy` — Standard classification accuracy (`accuracy_score`).  
- `balanced_acc` — **Balanced accuracy**, the average of per-class recall values (`balanced_accuracy_score`).  
- `macro_precision`, `macro_recall`, `macro_f1` — **Macro-averaged** metrics across all classes (`average="macro"`).  
- `micro_f1` — **Micro-averaged** F1 score over all instances (`average="micro"`).  
- `weighted_f1` — **Weighted F1**, averaging class-level F1 scores weighted by class support (`average="weighted"`).  
- `coverage_vs_gt` — Ratio of evaluated samples to total ground-truth samples.  
- `coverage_vs_pred` — Ratio of evaluated samples to total prediction files.  

Additional diagnostic statistics:  
- `n_gt_total`, `n_pred_files`, `n_eval_used`


---

## 🧠 Tasks Summary
Each task corresponds to one stage of IRAC reasoning in PTAB appeals:

1. **Issue Type (IRAC–Issue)**  
   Identify contested statutory grounds under 35 U.S.C. (§101/102/103/112/Others).

2. **Board Authorities (IRAC–Rule)**  
   Predict which procedural provisions under 37 C.F.R. (§ 41.50 variants, etc.) are cited by the Board.

3. **Subdecision (IRAC–Conclusion)**  
   Predict the Board’s final outcome (e.g., *Affirmed*, *Reversed*, *Affirmed-in-Part*).

---

## 📊 Evaluated Models
**Closed-source (commercial)**  
Claude-Sonnet-4 · Gemini-2.5-Pro · GPT-4o · GPT-o3 · Solar-Pro2  

**Open-source**  
LLaMA-3.1-8B-Instruct · Qwen-3-8B · Mistral-7B-Instruct · T5-Gemma-2B  

---

## 💾 Dataset Access
The full dataset, metadata, and opinion-split JSONs are available at  
👉 **[TeamLab/pilot-bench](To be updated)**

Each PTAB case is aligned with its corresponding USPTO patent and contains:
- `appellant_arguments`, `examiner_findings`, `ptab_opinion`
- standardized labels for Issue Type, Board Authorities, and Subdecision tasks  

---

## 🧮 Example Output
```json
{
    // Issue Type
  "issue_type": [
    "102",
    "103"
  ]
}
```
```json
{
    // Board Authorities
  "board_ruling": [
    "37 CFR 41.50",
    "37 CFR 41.50(a)"
  ]
}
```
```json
{
    // Subdecision (Fine-grained)
  "decision_number": 0,
  "decision_type": "Affirmed"
}
```
```json
{
    // Subdecision (Coarse-grained)
  "decision_type": "Reversed",
  "decision_number": 4
}
```
---

## 🧑‍⚖️ Citation
If you use this repository or dataset, please cite:
```bibtex
@inproceedings{jang2025pilotbench,
  title     = {PILOT-Bench: A Benchmark for Legal Reasoning in the Patent Domain with IRAC-Aligned Classification Tasks},
  author    = {Yehoon Jang and Chaewon Lee and Hyun-seok Min and Sungchul Choi},
  year      = {2025},
  booktitle = {Proceedings of the EMNLP 2025 (NLLP Workshop)},
  url       = {https://github.com/TeamLab/pilot-bench}
}
```

---

## ⚖️ License
Released under **CC BY 4.0** for research and educational purposes only.  
This repository and dataset **must not** be used to provide or automate legal advice, adjudication, or PTAB decision-making.

---

## 💬 Contact
For research inquiries or collaborations:
```
Yehoon Jang   : jangyh0420@pukyong.ac.kr  
Chaewon Lee   : oochaewon@pukyong.ac.kr
Hyun-seok Min : min6284@gmail.com  
Sungchul Choi : sc82.choi@pknu.ac.kr
```

---

## 🧩 Acknowledgments
This work was supported by  
- **National Research Foundation of Korea (NRF)** – Grant No. RS-2024-00354675 (70%)  
- **IITP (ICT Challenge and Advanced Network of HRD)** – Grant No. IITP-2023-RS-2023-00259806 (30%)  
under the supervision of the **Ministry of Science and ICT (MSIT), Korea**.
