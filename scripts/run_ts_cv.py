"""Expanding-window time-series CV: отчёт по средней точности (без подгонки под один фиксированный тест).

Пример:
  py scripts/run_ts_cv.py --config configs/modeling.yaml --splits 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit

from rf_project.config import load_yaml
from rf_project.modeling import _split_by_time, _target_series
from rf_project.io_utils import read_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", type=int, default=5)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    proc_dir = Path(cfg["paths"]["processed_dir"])
    df = read_table(proc_dir / "samples_modeling.parquet").sort_values("meeting_date")
    drop_cols = set(cfg["features"]["drop_cols"])
    X_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    X = df[X_cols].values
    y, y_desc = _target_series(df, cfg)

    tscv = TimeSeriesSplit(n_splits=args.splits)
    accs: list[float] = []
    f1w: list[float] = []
    f1m: list[float] = []
    bals: list[float] = []
    for tr_idx, te_idx in tscv.split(X):
        if len(np.unique(y.iloc[te_idx])) < 2:
            continue
        if len(np.unique(y.iloc[tr_idx])) < 2:
            continue
        m = LogisticRegression(
            max_iter=8000, class_weight="balanced", random_state=cfg["modeling"]["random_seed"]
        )
        m.fit(X[tr_idx], y.iloc[tr_idx])
        pred = m.predict(X[te_idx])
        ye = y.iloc[te_idx]
        accs.append(float(accuracy_score(ye, pred)))
        f1w.append(float(f1_score(ye, pred, average="weighted", zero_division=0)))
        f1m.append(float(f1_score(ye, pred, average="macro", zero_division=0)))
        bals.append(float(balanced_accuracy_score(ye, pred)))

    train, valid, test = _split_by_time(df, cfg["modeling"]["valid_start_date"], cfg["modeling"]["test_start_date"])
    report = {
        "primary_cv_metric": "fold_macro_f1",
        "target": y_desc,
        "time_series_cv_splits": args.splits,
        "fold_accuracy_mean": float(np.mean(accs)) if accs else None,
        "fold_accuracy_std": float(np.std(accs)) if accs else None,
        "fold_accuracy_each": accs,
        "fold_macro_f1_mean": float(np.mean(f1m)) if f1m else None,
        "fold_macro_f1_std": float(np.std(f1m)) if f1m else None,
        "fold_macro_f1_each": f1m,
        "fold_weighted_f1_mean": float(np.mean(f1w)) if f1w else None,
        "fold_weighted_f1_std": float(np.std(f1w)) if f1w else None,
        "fold_weighted_f1_each": f1w,
        "fold_balanced_accuracy_mean": float(np.mean(bals)) if bals else None,
        "fold_balanced_accuracy_std": float(np.std(bals)) if bals else None,
        "fold_balanced_accuracy_each": bals,
        "holdout_sizes": {"train": len(train), "valid": len(valid), "test": len(test)},
    }
    out_path = Path(cfg["paths"]["reports_dir"]) / "time_series_cv.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
