from __future__ import annotations

from pathlib import Path

import pandas as pd

from .embeddings import load_news_embedding_columns
from .io_utils import read_table, save_table


def _source_weight(source: str) -> float:
    s = str(source).lower()
    if s == "cbr":
        return 1.6
    if s in {"interfax", "kommersant", "vedomosti", "rbc", "tass"}:
        return 1.2
    return 1.0


def _half_trend(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 4:
        return 0.0
    mid = len(s) // 2
    a = float(s.iloc[:mid].mean())
    b = float(s.iloc[mid:].mean())
    return b - a


def _hold_streak_prior(classes: list[str]) -> list[int]:
    out: list[int] = []
    for i in range(len(classes)):
        if i == 0:
            out.append(0)
            continue
        s = 0
        j = i - 1
        while j >= 0 and classes[j] == "hold":
            s += 1
            j -= 1
        out.append(s)
    return out


def run_feature_build(cfg: dict) -> Path:
    raw_dir = Path(cfg["paths"]["raw_dir"])
    proc_dir = Path(cfg["paths"]["processed_dir"])
    interim = Path(cfg["paths"]["interim_dir"])
    window_days = int(cfg["dataset"]["window_days"])

    edges = read_table(proc_dir / "graph_edges.parquet")
    macro = read_table(raw_dir / "macro_raw.parquet")
    rates = read_table(raw_dir / "rates_raw.parquet")
    entities = read_table(interim / "entities.parquet") if (interim / "entities.parquet").exists() else pd.DataFrame()
    emb_df = load_news_embedding_columns(interim)
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")] if emb_df is not None else []

    edges["timestamp"] = pd.to_datetime(edges["timestamp"], utc=True)
    rates["meeting_date"] = pd.to_datetime(rates["meeting_date"], utc=True)
    macro["feature_date"] = pd.to_datetime(macro["feature_date"], utc=True)

    rates = rates.sort_values("meeting_date").reset_index(drop=True)
    cls_list = rates["target_class"].astype(str).tolist()
    rates["hold_streak_prior"] = _hold_streak_prior(cls_list)
    rates["prev_delta_bp"] = rates["delta_bp"].shift(1).fillna(0).astype(int)
    _ord = {"hold": 0, "up": 1, "down": 2}
    rates["prev_target_ord"] = rates["target_class"].shift(1).map(_ord).fillna(0.0)

    macro["feature_date"] = pd.to_datetime(macro["feature_date"]).dt.normalize()

    samples = []
    for _, m in rates.iterrows():
        t = m["meeting_date"]
        start = t - pd.Timedelta(days=window_days)
        edge_w = edges[(edges["timestamp"] >= start) & (edges["timestamp"] < t)]
        ts = pd.Timestamp(t).normalize()
        ss = pd.Timestamp(start).normalize()
        macro_w = macro[(macro["feature_date"] >= ss) & (macro["feature_date"] < ts)]
        if macro_w.empty or len(macro_w) < 3:
            macro_w = macro[macro["feature_date"] < ts].tail(max(window_days * 2, 30))
        if macro_w.empty:
            continue

        cpi_trend = _half_trend(macro_w["cpi_yoy"])
        usd_trend = _half_trend(macro_w["usd_rub"])

        news_ids = edge_w["news_id"].unique().tolist() if not edge_w.empty else []
        if not edge_w.empty:
            edge_w = edge_w.copy()
            days_to_meeting = (t - edge_w["timestamp"]).dt.total_seconds() / 86400.0
            edge_w["time_decay_w"] = (1.0 / (1.0 + days_to_meeting)).clip(lower=0.05, upper=1.0)
            edge_w["source_w"] = edge_w["source"].map(_source_weight).fillna(1.0)
            edge_w["effective_w"] = (
                edge_w["quality_weight"].fillna(1.0) * edge_w["time_decay_w"] * edge_w["source_w"]
                if "quality_weight" in edge_w.columns
                else edge_w["time_decay_w"] * edge_w["source_w"]
            )
        else:
            edge_w["effective_w"] = []
        snippet_count = int((edge_w["text_tier"] == "snippet_only").sum()) if "text_tier" in edge_w.columns else 0
        full_count = int((edge_w["text_tier"] == "full_text").sum()) if "text_tier" in edge_w.columns else int(len(edge_w))
        quality_weight_sum = (
            float(edge_w["quality_weight"].fillna(1.0).sum()) if "quality_weight" in edge_w.columns and not edge_w.empty else 0.0
        )
        effective_weight_sum = float(edge_w["effective_w"].sum()) if not edge_w.empty else 0.0
        ner_org = ner_per = ner_loc = ner_misc = ner_other = 0
        ew = pd.DataFrame()
        if not entities.empty and news_ids:
            ew = entities[entities["news_id"].isin(news_ids)]
            if not ew.empty and "entity_type" in ew.columns:
                for et, cnt in ew.groupby("entity_type").size().items():
                    u = str(et).upper()
                    if u == "ORG":
                        ner_org += int(cnt)
                    elif u == "PER":
                        ner_per += int(cnt)
                    elif u == "LOC":
                        ner_loc += int(cnt)
                    elif u in ("MISC", "GEOPOLIT", "MEDIA"):
                        ner_misc += int(cnt)
                    else:
                        ner_other += int(cnt)
        ner_unique = int(ew["canonical_id"].nunique()) if not ew.empty and "canonical_id" in ew.columns else 0

        row = {
            "sample_id": f"s_{t.date()}",
            "meeting_date": t,
            "window_start": start,
            "window_end": t,
            "rate_before": float(m["rate_before"]),
            "prev_delta_bp": int(m["prev_delta_bp"]),
            "prev_target_ord": float(m["prev_target_ord"]),
            "hold_streak_prior": int(m["hold_streak_prior"]),
            "edges_count": int(len(edge_w)),
            "edges_count_full_text": full_count,
            "edges_count_snippet_only": snippet_count,
            "edges_quality_weight_sum": quality_weight_sum,
            "edges_effective_weight_sum": effective_weight_sum,
            "unique_news_sources": int(edge_w["source"].nunique()) if not edge_w.empty else 0,
            "mentions_mean_conf": float(edge_w["model_confidence"].mean()) if not edge_w.empty else 0.0,
            "mentions_std_conf": float(edge_w["model_confidence"].std()) if len(edge_w) > 1 else 0.0,
            "cpi_mean": float(macro_w["cpi_yoy"].mean()),
            "cpi_std": float(macro_w["cpi_yoy"].std()) if len(macro_w) > 1 else 0.0,
            "cpi_trend_halves": float(cpi_trend),
            "usd_rub_mean": float(macro_w["usd_rub"].mean()),
            "usd_rub_std": float(macro_w["usd_rub"].std()) if len(macro_w) > 1 else 0.0,
            "usd_trend_halves": float(usd_trend),
            "brent_mean": float(macro_w["brent"].mean()),
            "rvi_mean": float(macro_w["rvi"].mean()),
            "pmi_mean": float(macro_w["pmi"].mean()),
            "ner_count_org": ner_org,
            "ner_count_per": ner_per,
            "ner_count_loc": ner_loc,
            "ner_count_misc": ner_misc,
            "ner_count_other": ner_other,
            "ner_unique_canonical": ner_unique,
            "target_class": m["target_class"],
            "target_delta_bp": m["delta_bp"],
        }

        if emb_df is not None and emb_cols and news_ids:
            sub = emb_df[emb_df["news_id"].isin(news_ids)]
            if len(sub) > 0:
                if "quality_weight" in edge_w.columns:
                    wmap = edge_w.drop_duplicates("news_id")[["news_id", "effective_w"]]
                    sub = sub.merge(wmap, on="news_id", how="left")
                    weights = sub["effective_w"].fillna(1.0).values
                    if float(weights.sum()) > 0:
                        vec = (sub[emb_cols].values * weights.reshape(-1, 1)).sum(axis=0) / float(weights.sum())
                    else:
                        vec = sub[emb_cols].values.mean(axis=0)
                else:
                    vec = sub[emb_cols].values.mean(axis=0)
                for i, c in enumerate(emb_cols):
                    row[c] = float(vec[i])
            else:
                for c in emb_cols:
                    row[c] = 0.0
        elif emb_cols:
            for c in emb_cols:
                row[c] = 0.0

        samples.append(row)

    out = pd.DataFrame(samples).sort_values("meeting_date").reset_index(drop=True)
    out_path = proc_dir / "samples_modeling.parquet"
    save_table(out, out_path)
    return out_path
