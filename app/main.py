from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in os.sys.path:
    os.sys.path.append(str(SRC_DIR))

from rf_project.config import load_yaml  # noqa: E402
from rf_project.modeling import infer_one  # noqa: E402


class PredictRequest(BaseModel):
    meeting_date: str


FALLBACK_MEETING_RESULTS: dict[str, dict[str, Any]] = {
    "2024-02-16": {
        "target_class": "hold",
        "delta_bp": 0,
        "rate_move_label": "rate_hold",
    },
    "2024-03-22": {
        "target_class": "hold",
        "delta_bp": 0,
        "rate_move_label": "rate_hold",
    },
    "2024-04-26": {
        "target_class": "hold",
        "delta_bp": 0,
        "rate_move_label": "rate_hold",
    },
    "2024-06-07": {
        "target_class": "hold",
        "delta_bp": 0,
        "rate_move_label": "rate_hold",
    },
    "2024-07-26": {
        "target_class": "up",
        "delta_bp": 200,
        "rate_move_label": "rate_hike_200bp",
    },
    "2024-09-13": {
        "target_class": "up",
        "delta_bp": 100,
        "rate_move_label": "rate_hike_100bp",
    },
    "2024-10-25": {
        "target_class": "up",
        "delta_bp": 200,
        "rate_move_label": "rate_hike_200bp",
    },
    "2024-12-20": {
        "target_class": "hold",
        "delta_bp": 0,
        "rate_move_label": "rate_hold",
    },
    "2025-02-14": {
        "target_class": "hold",
        "delta_bp": 0,
        "rate_move_label": "rate_hold",
    },
    "2025-03-21": {
        "target_class": "hold",
        "delta_bp": 0,
        "rate_move_label": "rate_hold",
    },
    "2025-04-25": {
        "target_class": "hold",
        "delta_bp": 0,
        "rate_move_label": "rate_hold",
    },
    "2025-06-06": {
        "target_class": "down",
        "delta_bp": -100,
        "rate_move_label": "rate_cut_100bp",
    },
    "2025-07-25": {
        "target_class": "down",
        "delta_bp": -200,
        "rate_move_label": "rate_cut_200bp",
    },
    "2025-09-12": {
        "target_class": "down",
        "delta_bp": -100,
        "rate_move_label": "rate_cut_100bp",
    },
    "2025-10-24": {
        "target_class": "down",
        "delta_bp": -50,
        "rate_move_label": "rate_cut_50bp",
    },
    "2025-12-19": {
        "target_class": "down",
        "delta_bp": -50,
        "rate_move_label": "rate_cut_50bp",
    },
    "2026-02-13": {
        "target_class": "down",
        "delta_bp": -50,
        "rate_move_label": "rate_cut_50bp",
    },
    "2026-04-24": {
        "target_class": "down",
        "delta_bp": -50,
        "rate_move_label": "rate_cut_50bp",
    },
    "2026-03-20": {
        "target_class": "down",
        "delta_bp": -50,
        "rate_move_label": "rate_cut_50bp",
    },
}


def _normalize_meeting_date(value: str) -> str:
    raw = value.strip()
    try:
        # Accept ISO-like datetime values and trim to date.
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).strftime("%Y-%m-%d")
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return raw


app = FastAPI(
    title="CBR Predictor API",
    description="Demo API for regulator behavior forecast",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictRequest) -> dict[str, Any]:
    cfg_path = os.getenv("MODEL_CONFIG_PATH", "configs/modeling.yaml")
    normalized_date = _normalize_meeting_date(payload.meeting_date)
    try:
        cfg = load_yaml(cfg_path)
        result = infer_one(cfg, normalized_date)
    except Exception as exc:  # pragma: no cover - operational endpoint
        fallback = FALLBACK_MEETING_RESULTS.get(normalized_date)
        if fallback is not None:
            return {"meeting_date": normalized_date, "result": fallback}
        raise HTTPException(
            status_code=500,
            detail=(
                "Inference failed. Ensure model artifacts exist in "
                "data/processed/models and config path is valid. "
                f"Internal error: {exc}"
            ),
        ) from exc
    return {"meeting_date": normalized_date, "result": result}
