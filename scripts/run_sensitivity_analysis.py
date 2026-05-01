from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from rf_project.config import load_yaml
from rf_project.modeling import evaluate_best, train_models
from rf_project.pipeline import run_make_dataset


def _run_scenario(base_cfg: dict, include_snippet: bool) -> dict:
    cfg = deepcopy(base_cfg)
    cfg["dataset"]["include_snippet"] = include_snippet
    label = "full_text_plus_snippet" if include_snippet else "full_text_only"
    run_make_dataset(cfg, version=f"sensitivity_{label}")
    train_models(cfg)
    ev = evaluate_best(cfg)
    return {
        "scenario": label,
        "include_snippet": include_snippet,
        "test_accuracy": ev.get("test_accuracy"),
        "test_balanced_accuracy": ev.get("test_balanced_accuracy"),
        "test_macro_f1": ev.get("test_macro_f1"),
        "test_weighted_f1": ev.get("test_weighted_f1"),
        "n_test": ev.get("n_test"),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-config", required=True)
    p.add_argument("--modeling-config", required=True)
    p.add_argument("--out-json", default="reports/sensitivity_analysis.json")
    args = p.parse_args()

    ds_cfg = load_yaml(args.dataset_config)
    mdl_cfg = load_yaml(args.modeling_config)
    cfg = deepcopy(ds_cfg)
    for k, v in mdl_cfg.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v

    full_only = _run_scenario(cfg, include_snippet=False)
    full_plus_snippet = _run_scenario(cfg, include_snippet=True)
    report = {
        "goal": "Sensitivity analysis for corpus coverage level",
        "scenarios": [full_only, full_plus_snippet],
        "delta_full_plus_minus_full_only": {
            "test_accuracy": (full_plus_snippet["test_accuracy"] or 0) - (full_only["test_accuracy"] or 0),
            "test_balanced_accuracy": (full_plus_snippet["test_balanced_accuracy"] or 0)
            - (full_only["test_balanced_accuracy"] or 0),
            "test_macro_f1": (full_plus_snippet["test_macro_f1"] or 0) - (full_only["test_macro_f1"] or 0),
            "test_weighted_f1": (full_plus_snippet["test_weighted_f1"] or 0) - (full_only["test_weighted_f1"] or 0),
        },
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
