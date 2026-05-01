from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def _split_by_time(df: pd.DataFrame, valid_start: str, test_start: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = df.sort_values("meeting_date").copy()
    x["meeting_date"] = pd.to_datetime(x["meeting_date"], utc=True)
    v = pd.to_datetime(valid_start, utc=True)
    t = pd.to_datetime(test_start, utc=True)
    return x[x["meeting_date"] < v], x[(x["meeting_date"] >= v) & (x["meeting_date"] < t)], x[x["meeting_date"] >= t]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default="data/processed/samples_modeling.parquet")
    p.add_argument("--valid-start", default="2022-05-01")
    p.add_argument("--test-start", default="2024-01-01")
    p.add_argument("--out-json", default="reports/two_stage_eval.json")
    args = p.parse_args()

    df = pd.read_parquet(args.samples).sort_values("meeting_date")
    drop_cols = {"sample_id", "meeting_date", "target_class", "target_delta_bp"}
    x_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    train, valid, test = _split_by_time(df, args.valid_start, args.test_start)

    y_change_tr = (train["target_class"] != "hold").astype(int)
    y_change_te = (test["target_class"] != "hold").astype(int)
    m_change = LogisticRegression(max_iter=12000, class_weight="balanced", random_state=42, C=0.3)
    m_change.fit(train[x_cols], y_change_tr)
    pred_change = m_change.predict(test[x_cols])

    train_dir = train[train["target_class"] != "hold"].copy()
    test_idx_change = test.index[pred_change == 1]
    m_dir = LogisticRegression(max_iter=12000, class_weight="balanced", random_state=42, C=0.3)
    m_dir.fit(train_dir[x_cols], train_dir["target_class"])

    pred_final = []
    for idx, row in test.iterrows():
        if idx in set(test_idx_change):
            pred_final.append(str(m_dir.predict(row[x_cols].to_frame().T)[0]))
        else:
            pred_final.append("hold")
    y_true = test["target_class"].astype(str)
    out = {
        "test_macro_f1": float(f1_score(y_true, pred_final, average="macro", zero_division=0)),
        "test_weighted_f1": float(f1_score(y_true, pred_final, average="weighted", zero_division=0)),
        "test_accuracy": float(accuracy_score(y_true, pred_final)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_true, pred_final)),
        "n_test": int(len(test)),
        "change_stage_test_f1": float(f1_score(y_change_te, pred_change, average="binary", zero_division=0)),
        "change_stage_test_accuracy": float(accuracy_score(y_change_te, pred_change)),
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
