from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .io_utils import read_table, save_table


def _normalize_text(text: str) -> str:
    return " ".join(str(text).replace("\n", " ").replace("\t", " ").split()).strip().lower()


def run_preprocess(cfg: dict) -> Path:
    raw_dir = Path(cfg["paths"]["raw_dir"])
    out_dir = Path(cfg["paths"]["interim_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_cfg = cfg.get("dataset") or {}
    min_len = int(ds_cfg.get("min_text_len", 120))
    full_text_min_len = int(ds_cfg.get("full_text_min_len", max(500, min_len)))
    snippet_min_len = int(ds_cfg.get("snippet_min_len", min_len))
    include_snippet = bool(ds_cfg.get("include_snippet", True))
    full_w = float(ds_cfg.get("full_text_weight", 1.0))
    snippet_w = float(ds_cfg.get("snippet_weight", 0.35))

    news = read_table(raw_dir / "news_raw.parquet")
    news["text_clean"] = news["text_raw"].fillna("").map(_normalize_text)
    news["title_clean"] = news["title"].fillna("").map(_normalize_text)
    news["text_len"] = news["text_clean"].str.len()
    news["lang"] = "ru"
    news["quality_flag"] = news["text_len"].map(lambda x: "ok" if x >= snippet_min_len else "short")
    news["text_tier"] = news["text_len"].map(
        lambda x: "full_text" if x >= full_text_min_len else ("snippet_only" if x >= snippet_min_len else "short")
    )
    news["quality_weight"] = news["text_tier"].map(
        {"full_text": full_w, "snippet_only": snippet_w}
    ).fillna(0.0)
    news["hash_exact"] = news["text_clean"].map(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest())
    news["is_duplicate_exact"] = news.duplicated("hash_exact", keep="first")
    news["is_duplicate_near"] = False
    news = news[~news["is_duplicate_exact"]].copy()
    news = news[news["quality_flag"] == "ok"].copy()
    if not include_snippet:
        news = news[news["text_tier"] == "full_text"].copy()

    out_path = out_dir / "news_clean.parquet"
    save_table(news, out_path)
    return out_path
