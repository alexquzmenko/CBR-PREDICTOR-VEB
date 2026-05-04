from __future__ import annotations

from pathlib import Path

from .config import ensure_dirs
from .embeddings import run_news_embeddings
from .features import run_feature_build
from .graph_build import run_graph_build
from .ingest import run_ingest
from .nlp import run_ner_re
from .preprocess import run_preprocess


def run_make_dataset(cfg: dict, version: str) -> dict:
    ensure_dirs(cfg["paths"]["raw_dir"], cfg["paths"]["interim_dir"], cfg["paths"]["processed_dir"], cfg["paths"]["reports_dir"])
    run_ingest(cfg)
    run_preprocess(cfg)
    run_ner_re(cfg)
    run_graph_build(cfg)
    run_news_embeddings(cfg)
    samples = run_feature_build(cfg)
    manifest = {
        "version": version,
        "samples_path": str(Path(samples)),
    }
    return manifest
