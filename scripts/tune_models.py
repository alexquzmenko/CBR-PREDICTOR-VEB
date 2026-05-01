from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from rf_project.config import load_yaml
from rf_project.io_utils import read_table
from rf_project.modeling import evaluate_best


def _merge_cfg(dataset_cfg: dict, modeling_cfg: dict) -> dict:
    cfg = copy.deepcopy(dataset_cfg)
    for k, v in modeling_cfg.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v
    return cfg


def _split_by_time(df: pd.DataFrame, valid_start: str, test_start: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = df.sort_values("meeting_date").copy()
    x["meeting_date"] = pd.to_datetime(x["meeting_date"], utc=True)
    v = pd.to_datetime(valid_start, utc=True)
    t = pd.to_datetime(test_start, utc=True)
    train = x[x["meeting_date"] < v]
    valid = x[(x["meeting_date"] >= v) & (x["meeting_date"] < t)]
    test = x[x["meeting_date"] >= t]
    return train, valid, test


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-config", required=True)
    p.add_argument("--modeling-config", required=True)
    p.add_argument("--out-json", default="reports/model_tuning.json")
    args = p.parse_args()

    ds = load_yaml(args.dataset_config)
    md = load_yaml(args.modeling_config)
    cfg = _merge_cfg(ds, md)

    proc_dir = Path(cfg["paths"]["processed_dir"])
    save_dir = Path(cfg["modeling"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    df = read_table(proc_dir / "samples_modeling.parquet")
    drop_cols = set(cfg["features"]["drop_cols"])
    x_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    train, valid, _ = _split_by_time(df, cfg["modeling"]["valid_start_date"], cfg["modeling"]["test_start_date"])
    y_col = cfg["modeling"]["target_col"]
    y_train = train[y_col]
    y_valid = valid[y_col]

    seed = int(cfg["modeling"]["random_seed"])
    candidates: list[tuple[str, dict, object]] = [
        ("logreg", {"C": 0.1, "class_weight": "balanced"}, LogisticRegression(max_iter=12000, random_state=seed, C=0.1, class_weight="balanced")),
        ("logreg", {"C": 0.5, "class_weight": "balanced"}, LogisticRegression(max_iter=12000, random_state=seed, C=0.5, class_weight="balanced")),
        ("logreg", {"C": 1.0, "class_weight": "balanced"}, LogisticRegression(max_iter=12000, random_state=seed, C=1.0, class_weight="balanced")),
        ("rf", {"n_estimators": 500, "max_depth": 6}, RandomForestClassifier(n_estimators=500, random_state=seed, class_weight="balanced_subsample", max_depth=6, min_samples_leaf=2)),
        ("rf", {"n_estimators": 700, "max_depth": 8}, RandomForestClassifier(n_estimators=700, random_state=seed, class_weight="balanced_subsample", max_depth=8, min_samples_leaf=2)),
        ("rf", {"n_estimators": 900, "max_depth": 10}, RandomForestClassifier(n_estimators=900, random_state=seed, class_weight="balanced_subsample", max_depth=10, min_samples_leaf=1)),
    ]

    rows = []
    best = None
    for name, params, model in candidates:
        model.fit(train[x_cols], y_train)
        pred = model.predict(valid[x_cols])
        m = {
            "model": name,
            "params": params,
            "valid_macro_f1": float(f1_score(y_valid, pred, average="macro", zero_division=0)),
            "valid_weighted_f1": float(f1_score(y_valid, pred, average="weighted", zero_division=0)),
            "valid_accuracy": float(accuracy_score(y_valid, pred)),
            "valid_balanced_accuracy": float(balanced_accuracy_score(y_valid, pred)),
        }
        rows.append(m)
        key = (m["valid_macro_f1"], m["valid_balanced_accuracy"], m["valid_accuracy"])
        if best is None or key > (
            best["valid_macro_f1"],
            best["valid_balanced_accuracy"],
            best["valid_accuracy"],
        ):
            best = {**m, "_model_obj": model}

    assert best is not None
    best_name = str(best["model"])
    joblib.dump(best["_model_obj"], save_dir / f"{best_name}.joblib")
    (save_dir / "best_model.txt").write_text(best_name, encoding="utf-8")

    # Reuse existing evaluator for test metrics with selected model.
    test_eval = evaluate_best(cfg)

    report = {
        "candidates": rows,
        "best_on_valid": {k: v for k, v in best.items() if k != "_model_obj"},
        "test_metrics_after_tuning": {
            "test_macro_f1": test_eval.get("test_macro_f1"),
            "test_weighted_f1": test_eval.get("test_weighted_f1"),
            "test_accuracy": test_eval.get("test_accuracy"),
            "test_balanced_accuracy": test_eval.get("test_balanced_accuracy"),
        },
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
