from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default="data/processed/samples_modeling.parquet")
    p.add_argument("--out-json", default="reports/nested_walkforward.json")
    p.add_argument("--splits", type=int, default=5)
    args = p.parse_args()

    df = pd.read_parquet(args.samples).sort_values("meeting_date")
    drop_cols = {"sample_id", "meeting_date", "target_class", "target_delta_bp"}
    x_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    X = df[x_cols].values
    y = df["target_class"].astype(str).values

    grid = [0.05, 0.1, 0.3, 0.5, 1.0, 2.0]
    tscv = TimeSeriesSplit(n_splits=args.splits)
    folds = []
    for tr_idx, te_idx in tscv.split(X):
        if len(np.unique(y[tr_idx])) < 2 or len(np.unique(y[te_idx])) < 2:
            continue
        inner = TimeSeriesSplit(n_splits=3)
        best_c, best_s = 1.0, -1.0
        for c in grid:
            scores = []
            for i_tr, i_va in inner.split(X[tr_idx]):
                y_tr = y[tr_idx][i_tr]
                y_va = y[tr_idx][i_va]
                if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
                    continue
                m = LogisticRegression(max_iter=12000, class_weight="balanced", C=c, random_state=42)
                m.fit(X[tr_idx][i_tr], y_tr)
                p_va = m.predict(X[tr_idx][i_va])
                scores.append(f1_score(y_va, p_va, average="macro", zero_division=0))
            s = float(np.mean(scores)) if scores else -1.0
            if s > best_s:
                best_s, best_c = s, c
        m = LogisticRegression(max_iter=12000, class_weight="balanced", C=best_c, random_state=42)
        m.fit(X[tr_idx], y[tr_idx])
        p_te = m.predict(X[te_idx])
        folds.append(
            {
                "best_c": best_c,
                "macro_f1": float(f1_score(y[te_idx], p_te, average="macro", zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(y[te_idx], p_te)),
            }
        )
    out = {
        "folds": folds,
        "macro_f1_mean": float(np.mean([f["macro_f1"] for f in folds])) if folds else None,
        "balanced_accuracy_mean": float(np.mean([f["balanced_accuracy"] for f in folds])) if folds else None,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
