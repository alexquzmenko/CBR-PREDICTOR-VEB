"""Сводный JSON для главы результатов: метрики hold-out, CV, валидация, ограничения.

  py scripts/generate_thesis_report.py --config configs/modeling.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from rf_project.config import load_yaml

LIMITATIONS_RU = (
    "Выборка ограничена числом заседаний и доступностью макро/новостей; временные сплиты дают малый тест. "
    "Режимы ДКП меняются (ужесточение/смягчение), поэтому метрики на одном отрезке шумные. "
    "Окно признаков использует только данные строго до даты заседания (без утечки будущего текста). "
    "Эмбеддинги и NER зависят от качества корпуса; при отключённых эмбеддингах колонки emb_* отсутствуют или нули. "
    "Покрытие корпуса неоднородно по месяцам и источникам, поэтому обязательно проверяется sensitivity: full_text only vs full_text+snippet."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    reports_dir = Path(cfg["paths"]["reports_dir"])
    models_dir = Path(cfg["modeling"]["save_dir"])

    eval_path = reports_dir / "eval_test.json"
    cv_path = reports_dir / "time_series_cv.json"
    valid_path = models_dir / "metrics_valid.json"
    meta_path = models_dir / "training_meta.json"
    sens_path = reports_dir / "sensitivity_analysis.json"
    checkpoint_path = reports_dir / "corpus_checkpoint.csv"

    summary: dict = {
        "primary_metric_for_thesis": "test_macro_f1 (и отчёт по классам в classification_report)",
        "limitations_ru": LIMITATIONS_RU,
    }

    if eval_path.exists():
        summary["holdout_eval"] = json.loads(eval_path.read_text(encoding="utf-8"))
    if cv_path.exists():
        summary["time_series_cross_validation"] = json.loads(cv_path.read_text(encoding="utf-8"))
    if valid_path.exists():
        summary["validation_tuning"] = json.loads(valid_path.read_text(encoding="utf-8"))
    if meta_path.exists():
        summary["training_meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
    if sens_path.exists():
        summary["sensitivity_analysis"] = json.loads(sens_path.read_text(encoding="utf-8"))
    if checkpoint_path.exists():
        cp = __import__("pandas").read_csv(checkpoint_path)
        summary["coverage_summary"] = {
            "batches_with_data": int(len(cp)),
            "rows_total_from_batches": int(cp["rows"].sum()) if "rows" in cp.columns else 0,
            "rows_full_text_from_batches": int(cp["full_text_rows"].sum()) if "full_text_rows" in cp.columns else 0,
            "rows_snippet_from_batches": int(cp["snippet_rows"].sum()) if "snippet_rows" in cp.columns else 0,
        }

    out_path = reports_dir / "thesis_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
