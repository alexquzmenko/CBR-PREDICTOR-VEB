from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
)
from sklearn.linear_model import Ridge

from .io_utils import read_table


def _baselines(
    y: pd.Series,
    cfg: dict,
    df_full: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, float | None]:
    if y.empty:
        return {}
    vc = y.value_counts(normalize=True)
    out: dict[str, float | None] = {"majority_class_accuracy": float(vc.max())}
    mode = (cfg.get("modeling") or {}).get("target_mode", "three_class")
    full = df_full.sort_values("meeting_date").copy()
    if mode == "binary_change":
        full["persist_pred"] = (full["target_delta_bp"].abs() > 0).astype(int).shift(1)
    else:
        tc = cfg["modeling"]["target_col"]
        full["persist_pred"] = full[tc].shift(1)
    merged = test_df.merge(full[["meeting_date", "persist_pred"]], on="meeting_date", how="left")
    mask = merged["persist_pred"].notna()
    if not mask.any():
        out["persistence_accuracy"] = None
        return out
    if mode == "binary_change":
        y_bin = (merged["target_delta_bp"].abs() > 0).astype(int)
        pred = merged["persist_pred"].astype(int)
    else:
        tc = cfg["modeling"]["target_col"]
        y_bin = merged[tc].astype(str)
        pred = merged["persist_pred"].astype(str)
    out["persistence_accuracy"] = float((y_bin[mask] == pred[mask]).mean())
    return out


def _selection_score(metric_name: str, f1_macro: float, f1_weighted: float, acc: float) -> float:
    if metric_name == "weighted_f1":
        return f1_weighted
    if metric_name == "combo_macro_acc":
        return 0.5 * f1_macro + 0.5 * acc
    return f1_macro


def _target_series(df: pd.DataFrame, cfg: dict) -> tuple[pd.Series, str]:
    """Returns y and label describing target (for metadata)."""
    mode = (cfg.get("modeling") or {}).get("target_mode", "three_class")
    if mode == "binary_change":
        y = (df["target_delta_bp"].abs() > 0).astype(int)
        return y, "binary_change"
    col = cfg["modeling"]["target_col"]
    return df[col], f"multiclass:{col}"


def _split_by_time(df: pd.DataFrame, valid_start: str, test_start: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("meeting_date").copy()
    df["meeting_date"] = pd.to_datetime(df["meeting_date"], utc=True)
    valid_start_ts = pd.to_datetime(valid_start, utc=True)
    test_start_ts = pd.to_datetime(test_start, utc=True)
    train = df[df["meeting_date"] < valid_start_ts]
    valid = df[(df["meeting_date"] >= valid_start_ts) & (df["meeting_date"] < test_start_ts)]
    test = df[df["meeting_date"] >= test_start_ts]
    return train, valid, test


def train_models(cfg: dict) -> dict:
    proc_dir = Path(cfg["paths"]["processed_dir"])
    save_dir = Path(cfg["modeling"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(proc_dir / "samples_modeling.parquet")
    drop_cols = set(cfg["features"]["drop_cols"])
    X_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]

    train, valid, test = _split_by_time(df, cfg["modeling"]["valid_start_date"], cfg["modeling"]["test_start_date"])
    if train.empty or valid.empty or test.empty:
        raise ValueError("Not enough data in one of train/valid/test splits. Adjust date boundaries.")

    y_train, y_train_desc = _target_series(train, cfg)
    y_valid, _ = _target_series(valid, cfg)

    seed = cfg["modeling"]["random_seed"]
    models = {
        "logreg": LogisticRegression(max_iter=8000, random_state=seed, class_weight="balanced"),
        "rf": RandomForestClassifier(
            n_estimators=400,
            random_state=seed,
            class_weight="balanced_subsample",
            max_depth=7,
            min_samples_leaf=2,
        ),
    }

    metrics = {}
    best_name = None
    best_score = -1.0
    sel_metric = (cfg.get("modeling") or {}).get("selection_metric", "macro_f1")
    for name, model in models.items():
        model.fit(train[X_cols], y_train)
        pred = model.predict(valid[X_cols])
        f1_macro = f1_score(y_valid, pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_valid, pred, average="weighted", zero_division=0)
        acc = accuracy_score(y_valid, pred)
        metrics[name] = {
            "valid_macro_f1": float(f1_macro),
            "valid_weighted_f1": float(f1_weighted),
            "valid_acc": float(acc),
            "selection_score": float(_selection_score(sel_metric, f1_macro, f1_weighted, acc)),
        }
        joblib.dump(model, save_dir / f"{name}.joblib")
        score = _selection_score(sel_metric, f1_macro, f1_weighted, acc)
        if score > best_score:
            best_score = score
            best_name = name

    with (save_dir / "metrics_valid.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with (save_dir / "best_model.txt").open("w", encoding="utf-8") as f:
        f.write(best_name or "rf")
    meta = {
        "target_mode": (cfg.get("modeling") or {}).get("target_mode", "three_class"),
        "target_description": y_train_desc,
        "selection_metric": sel_metric,
        "best_model": best_name,
        "valid_start_date": cfg["modeling"]["valid_start_date"],
        "test_start_date": cfg["modeling"]["test_start_date"],
    }
    with (save_dir / "training_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return {"best_model": best_name, "metrics": metrics}


def _delta_bp_to_class(v: float) -> str:
    if v > 12.5:
        return "up"
    if v < -12.5:
        return "down"
    return "hold"


def _delta_regression_eval(cfg: dict, df: pd.DataFrame, X_cols: list[str]) -> dict | None:
    if not (cfg.get("modeling") or {}).get("report_delta_regression", False):
        return None
    mode = (cfg.get("modeling") or {}).get("target_mode", "three_class")
    if mode != "three_class":
        return None
    train, valid, test = _split_by_time(df, cfg["modeling"]["valid_start_date"], cfg["modeling"]["test_start_date"])
    if train.empty or valid.empty or test.empty:
        return None
    reg = Ridge(alpha=2.0)
    reg.fit(train[X_cols], train["target_delta_bp"])
    pred_v = reg.predict(valid[X_cols])
    pred_t = reg.predict(test[X_cols])

    mae_v = float(mean_absolute_error(valid["target_delta_bp"], pred_v))
    mae_t = float(mean_absolute_error(test["target_delta_bp"], pred_t))
    cls_v_pred = [_delta_bp_to_class(x) for x in pred_v]
    cls_t_pred = [_delta_bp_to_class(x) for x in pred_t]
    f1_v = float(f1_score(valid["target_class"], cls_v_pred, average="macro", zero_division=0))
    f1_t = float(f1_score(test["target_class"], cls_t_pred, average="macro", zero_division=0))
    return {
        "ridge_valid_mae_delta_bp": mae_v,
        "ridge_test_mae_delta_bp": mae_t,
        "ridge_valid_macro_f1_mapped_class": f1_v,
        "ridge_test_macro_f1_mapped_class": f1_t,
    }


def evaluate_best(cfg: dict) -> dict:
    proc_dir = Path(cfg["paths"]["processed_dir"])
    save_dir = Path(cfg["modeling"]["save_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(proc_dir / "samples_modeling.parquet")
    drop_cols = set(cfg["features"]["drop_cols"])
    X_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]

    _, _, test = _split_by_time(df, cfg["modeling"]["valid_start_date"], cfg["modeling"]["test_start_date"])
    best_name = (save_dir / "best_model.txt").read_text(encoding="utf-8").strip()
    model = joblib.load(save_dir / f"{best_name}.joblib")

    y_test, _ = _target_series(test, cfg)
    X_test = test[X_cols]
    pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, pred))
    base = _baselines(y_test, cfg, df, test)
    maj = float(base.get("majority_class_accuracy") or 0.0)
    pers = base.get("persistence_accuracy")
    labels_cm = sorted(
        set(pd.Series(y_test).astype(str).unique().tolist()) | set(pd.Series(pred).astype(str).unique().tolist())
    )
    cm = confusion_matrix(pd.Series(y_test).astype(str), pd.Series(pred).astype(str), labels=labels_cm)
    result = {
        "primary_metric": "test_macro_f1",
        "model": best_name,
        "test_accuracy": acc,
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "test_macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "test_weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        "confusion_matrix": {"labels": labels_cm, "matrix": cm.tolist()},
        "n_test": int(len(test)),
        "class_distribution_test": {str(k): int(v) for k, v in pd.Series(y_test).value_counts().items()},
        "baselines": base,
        "vs_baselines": {
            "beats_majority_accuracy": acc > maj,
            "beats_persistence_accuracy": pers is not None and acc > float(pers),
        },
    }
    extra = _delta_regression_eval(cfg, df, X_cols)
    if extra:
        result["delta_regression_auxiliary"] = extra
    if hasattr(model, "predict_proba"):
        try:
            result["test_log_loss"] = float(
                log_loss(y_test, model.predict_proba(X_test), labels=model.classes_)
            )
        except (ValueError, TypeError, AttributeError):
            pass
    with (reports_dir / "eval_test.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def infer_one(cfg: dict, meeting_date: str) -> dict:
    proc_dir = Path(cfg["paths"]["processed_dir"])
    save_dir = Path(cfg["modeling"]["save_dir"])
    df = read_table(proc_dir / "samples_modeling.parquet")
    df["meeting_date"] = pd.to_datetime(df["meeting_date"], utc=True)
    row = df[df["meeting_date"] == pd.to_datetime(meeting_date, utc=True)]
    if row.empty:
        raise ValueError(f"No sample found for meeting_date={meeting_date}")

    drop_cols = set(cfg["features"]["drop_cols"])
    X_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    best_name = (save_dir / "best_model.txt").read_text(encoding="utf-8").strip()
    model = joblib.load(save_dir / f"{best_name}.joblib")
    pred = model.predict(row[X_cols])[0]
    meta_path = save_dir / "training_meta.json"
    out: dict = {"meeting_date": meeting_date, "model": best_name, "prediction": str(pred)}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        out["target_mode"] = meta.get("target_mode", "three_class")
        if meta.get("target_mode") == "binary_change":
            out["prediction_label"] = "change" if int(pred) == 1 else "hold"
    return out
