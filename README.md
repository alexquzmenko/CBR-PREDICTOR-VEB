# Regulator Forecast Project

End-to-end prototype for forecasting Bank of Russia key-rate decisions from:
- news semantics,
- temporal relation graph,
- macro/market factors.

## Demo metrics (for supervisor preview only)

The numbers below are demo targets used to present expected project outputs and UI/report structure.
They are not final defended results and must be replaced by fully reproduced experiment runs before pre-defense.

| Model | Macro-F1 | ROC-AUC (ovr) | Balanced Accuracy | Brier Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.48 | 0.66 | 0.52 | 0.21 |
| XGBoost | 0.51 | 0.69 | 0.56 | 0.20 |
| BiLSTM | 0.54 | 0.72 | 0.58 | 0.19 |
| Transformer-only | 0.58 | 0.76 | 0.62 | 0.17 |
| Graph-only | 0.56 | 0.74 | 0.60 | 0.18 |
| **Hybrid (Text + GNN + Macro)** | **0.75** | **0.89** | **0.78** | **0.11** |

Relative Macro-F1 uplift vs strongest baseline: approximately `+29.3%` (about `+30%`).

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python scripts/run_make_dataset.py --config configs/dataset.yaml --version v1.0.0
python scripts/run_train.py --config configs/modeling.yaml
python scripts/run_eval.py --config configs/modeling.yaml
python scripts/run_infer.py --config configs/modeling.yaml --meeting-date 2025-12-20
```

## Pipeline
1. `ingest`: load raw news + macro + rate decisions (demo fallback generator included).
2. `preprocess`: clean text, exact dedup, quality flags.
3. `ner_re`: lightweight rule-based extraction (replaceable by RuBERT models later).
4. `graph`: temporal edge build + node stats.
5. `features`: meeting-window graph + macro features.
6. `modeling`: baseline classifiers + hybrid model.

## Data contract
Generated artifacts are placed in:
- `data/raw`
- `data/interim`
- `data/processed`

Main modeling table:
- `data/processed/samples_modeling.parquet`

## Notes
- This is a runnable baseline implementation for thesis development.
- You can swap rule-based NER/RE with transformer pipelines while keeping IO contracts unchanged.
