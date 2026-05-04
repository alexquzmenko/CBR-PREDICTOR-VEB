from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io_utils import read_table, save_table


def run_graph_build(cfg: dict) -> tuple[Path, Path]:
    interim = Path(cfg["paths"]["interim_dir"])
    processed = Path(cfg["paths"]["processed_dir"])
    processed.mkdir(parents=True, exist_ok=True)

    entities = read_table(interim / "entities.parquet")
    relations = read_table(interim / "relations.parquet")
    news = read_table(interim / "news_clean.parquet")
    keep_cols = ["news_id", "published_at_utc", "source", "text_tier", "quality_weight"]
    for c in keep_cols:
        if c not in news.columns:
            if c == "text_tier":
                news[c] = "full_text"
            elif c == "quality_weight":
                news[c] = 1.0
            else:
                news[c] = ""
    news = news[keep_cols]

    edges = relations.merge(news, on="news_id", how="left")
    edges["timestamp"] = pd.to_datetime(edges["event_time"])
    edges["source_count"] = 1
    edges_path = processed / "graph_edges.parquet"
    save_table(edges, edges_path)

    nodes = (
        entities.groupby(["canonical_id", "entity_type"], as_index=False)
        .agg(mentions=("entity_id", "count"))
        .rename(columns={"canonical_id": "node_id"})
    )
    nodes_path = processed / "graph_nodes.parquet"
    save_table(nodes, nodes_path)
    return nodes_path, edges_path
