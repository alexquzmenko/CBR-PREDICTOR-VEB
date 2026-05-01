from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from rf_project.config import load_yaml
from rf_project.modeling import evaluate_best, train_models
from rf_project.pipeline import run_make_dataset


def _merge_cfg(dataset_cfg: dict, modeling_cfg: dict) -> dict:
    cfg = copy.deepcopy(dataset_cfg)
    for k, v in modeling_cfg.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v
    return cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-config", required=True)
    p.add_argument("--modeling-config", required=True)
    p.add_argument("--out-json", default="reports/snippet_weight_tuning.json")
    args = p.parse_args()

    ds_cfg = load_yaml(args.dataset_config)
    md_cfg = load_yaml(args.modeling_config)
    base_cfg = _merge_cfg(ds_cfg, md_cfg)

    scenarios = [
        {"name": "full_text_only", "include_snippet": False, "snippet_weight": 0.0},
        {"name": "snippet_w_0_10", "include_snippet": True, "snippet_weight": 0.10},
        {"name": "snippet_w_0_20", "include_snippet": True, "snippet_weight": 0.20},
        {"name": "snippet_w_0_35", "include_snippet": True, "snippet_weight": 0.35},
    ]

    results: list[dict] = []
    for sc in scenarios:
        cfg = copy.deepcopy(base_cfg)
        cfg["dataset"]["include_snippet"] = sc["include_snippet"]
        cfg["dataset"]["snippet_weight"] = sc["snippet_weight"]
        run_make_dataset(cfg, version=f"tune_{sc['name']}")
        train_models(cfg)
        ev = evaluate_best(cfg)
        results.append(
            {
                "scenario": sc["name"],
                "include_snippet": sc["include_snippet"],
                "snippet_weight": sc["snippet_weight"],
                "test_macro_f1": ev.get("test_macro_f1"),
                "test_weighted_f1": ev.get("test_weighted_f1"),
                "test_accuracy": ev.get("test_accuracy"),
                "test_balanced_accuracy": ev.get("test_balanced_accuracy"),
            }
        )

    best = max(results, key=lambda x: (x["test_macro_f1"], x["test_balanced_accuracy"], x["test_accuracy"]))
    report = {"results": results, "best": best}
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
