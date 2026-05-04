from __future__ import annotations

import os
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
    try:
        cfg = load_yaml(cfg_path)
        result = infer_one(cfg, payload.meeting_date)
    except Exception as exc:  # pragma: no cover - operational endpoint
        raise HTTPException(
            status_code=500,
            detail=(
                "Inference failed. Ensure model artifacts exist in "
                "data/processed/models and config path is valid. "
                f"Internal error: {exc}"
            ),
        ) from exc
    return {"meeting_date": payload.meeting_date, "result": result}
