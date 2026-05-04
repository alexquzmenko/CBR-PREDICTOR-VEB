from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from .io_utils import read_table, save_table

ENTITY_DICT = {
    "банк россии": "ORG",
    "цб": "ORG",
    "инфляц": "IND",
    "ключев": "IND",
    "рубл": "CUR",
}


def _nlp_cfg(cfg: dict) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "ner_model_id": "viktoroo/sberbank-rubert-base-collection3",
        "ner_chunk_chars": 1200,
        "ner_chunk_stride": 300,
        "ner_batch_size": 4,
        "ner_device": -1,
        "max_entities_for_relations": 36,
    }
    merged = {**defaults, **(cfg.get("nlp") or {})}
    return merged


def _chunk_text(text: str, window: int, stride: int) -> list[tuple[str, int]]:
    text = text or ""
    if len(text) <= window:
        return [(text, 0)]
    chunks: list[tuple[str, int]] = []
    pos = 0
    while pos < len(text):
        chunk = text[pos : pos + window]
        chunks.append((chunk, pos))
        if pos + window >= len(text):
            break
        pos += stride
    return chunks


def _canonical_id(entity_text: str) -> str:
    s = entity_text.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w\-]", "", s, flags=re.UNICODE)
    return (s[:200] if s else "entity")


def _normalize_entity_group(label: str) -> str:
    if not label:
        return "MISC"
    if label.startswith("B-") or label.startswith("I-"):
        return label[2:] or "MISC"
    return label


def _dedupe_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep highest score per identical (start, end, type)."""
    best: dict[tuple[int, int, str], dict[str, Any]] = {}
    for sp in spans:
        key = (int(sp["span_start"]), int(sp["span_end"]), str(sp["entity_type"]))
        prev = best.get(key)
        if prev is None or float(sp["model_confidence"]) > float(prev["model_confidence"]):
            best[key] = sp
    return sorted(best.values(), key=lambda x: (x["span_start"], x["span_end"]))


def _merge_overlapping(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve overlaps between adjacent windows: prefer higher confidence on conflict."""
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: (x["span_start"], -x["span_end"]))
    out: list[dict[str, Any]] = []
    for sp in spans:
        if not out:
            out.append(sp)
            continue
        last = out[-1]
        overlap = sp["span_start"] < last["span_end"] and sp["span_end"] > last["span_start"]
        if overlap and sp["entity_type"] == last["entity_type"]:
            if float(sp["model_confidence"]) > float(last["model_confidence"]):
                out[-1] = sp
            continue
        if overlap and sp["entity_type"] != last["entity_type"]:
            if float(sp["model_confidence"]) > float(last["model_confidence"]):
                out[-1] = sp
            continue
        out.append(sp)
    return out


def _extract_spans_for_text(
    text: str,
    ner,
    chunk_chars: int,
    chunk_stride: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    chunks = _chunk_text(text, chunk_chars, chunk_stride)
    raw_spans: list[dict[str, Any]] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c[0] for c in batch]
        offsets = [c[1] for c in batch]
        preds = ner(texts, batch_size=len(texts))
        for offset, doc_preds in zip(offsets, preds):
            if not doc_preds:
                continue
            for p in doc_preds:
                start = int(p["start"]) + offset
                end = int(p["end"]) + offset
                etype = _normalize_entity_group(str(p.get("entity_group", "MISC")))
                score = float(p.get("score", 0.0))
                raw_spans.append(
                    {
                        "entity_text": text[start:end],
                        "entity_type": etype,
                        "span_start": start,
                        "span_end": end,
                        "model_confidence": score,
                    }
                )
    merged = _dedupe_spans(raw_spans)
    return _merge_overlapping(merged)


def run_ner_re(cfg: dict) -> tuple[Path, Path]:
    nc = _nlp_cfg(cfg)
    interim = Path(cfg["paths"]["interim_dir"])
    news = read_table(interim / "news_clean.parquet")
    ner = None
    try:
        from transformers import pipeline

        ner = pipeline(
            "token-classification",
            model=nc["ner_model_id"],
            tokenizer=nc["ner_model_id"],
            aggregation_strategy="simple",
            device=int(nc["ner_device"]),
        )
    except Exception as e:
        print(f"NER fallback to rule-based extractor: {e}", flush=True)

    entities_rows: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    max_rel = max(2, int(nc["max_entities_for_relations"]))

    for n_total, (_, row) in enumerate(news.iterrows(), start=1):
        text = str(row["text_clean"])
        if ner is not None:
            spans = _extract_spans_for_text(
                text,
                ner,
                int(nc["ner_chunk_chars"]),
                int(nc["ner_chunk_stride"]),
                int(nc["ner_batch_size"]),
            )
        else:
            spans = []
            lt = text.lower()
            for key, et in ENTITY_DICT.items():
                m = re.search(key, lt)
                if m:
                    spans.append(
                        {
                            "entity_text": text[m.start() : m.end()],
                            "entity_type": et,
                            "span_start": m.start(),
                            "span_end": m.end(),
                            "model_confidence": 0.6,
                        }
                    )
        spans = sorted(spans, key=lambda x: (x["span_start"], x["span_end"]))

        for idx, sp in enumerate(spans):
            entities_rows.append(
                {
                    "entity_id": f"{row['news_id']}_e_{idx}",
                    "news_id": row["news_id"],
                    "entity_text": sp["entity_text"],
                    "entity_type": sp["entity_type"],
                    "span_start": sp["span_start"],
                    "span_end": sp["span_end"],
                    "model_confidence": sp["model_confidence"],
                    "canonical_id": _canonical_id(sp["entity_text"]),
                }
            )

        order = sorted(range(len(spans)), key=lambda i: -float(spans[i]["model_confidence"]))
        rel_ix = order[: min(max_rel, len(spans))]
        rel_ix.sort()
        for a in range(len(rel_ix)):
            for b in range(a + 1, len(rel_ix)):
                i, j = rel_ix[a], rel_ix[b]
                relation_rows.append(
                    {
                        "relation_id": f"{row['news_id']}_r_{i}_{j}",
                        "news_id": row["news_id"],
                        "head_entity_id": f"{row['news_id']}_e_{i}",
                        "tail_entity_id": f"{row['news_id']}_e_{j}",
                        "relation_type": "cooccurs",
                        "model_confidence": float(
                            (spans[i]["model_confidence"] + spans[j]["model_confidence"]) / 2.0
                        ),
                        "sentiment_score": 0.0,
                        "event_time": row["published_at_utc"],
                    }
                )
        if n_total % 50 == 0:
            print(f"NER: processed {n_total} documents", flush=True)

    entities = pd.DataFrame(entities_rows)
    relations = pd.DataFrame(relation_rows)
    ent_path = interim / "entities.parquet"
    rel_path = interim / "relations.parquet"
    save_table(entities, ent_path)
    save_table(relations, rel_path)
    return ent_path, rel_path
