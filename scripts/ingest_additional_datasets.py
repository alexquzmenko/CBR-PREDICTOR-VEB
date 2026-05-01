from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

THEMATIC_MUST_ANY = [
    "ключев",
    "ставк",
    "инфляц",
    "денежно-кредит",
    "банк россии",
    "цб",
    "дкп",
    "офз",
    "доходност",
    "кредит",
]


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "cp1251", "windows-1251", "utf-8-sig"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)


def _to_iso(ts: pd.Series) -> pd.Series:
    x = pd.to_datetime(ts, errors="coerce", utc=True)
    return x.dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_news_id(source: str, text: str) -> str:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{source}_{h}"


def _normalize_source(x: pd.Series) -> pd.Series:
    s = x.fillna("").astype(str).str.lower()
    s = s.str.replace(r"^the\s+", "", regex=True)
    s = s.where(s.str.contains(r"[a-z]", regex=True), "dataset_external")
    s = s.where(s.str.fullmatch(r"[a-z0-9_\\-]+"), "dataset_external")
    return s


def _is_thematic(title: pd.Series, text: pd.Series) -> pd.Series:
    blob = (title.fillna("").astype(str) + " " + text.fillna("").astype(str)).str.lower()
    mask = pd.Series(False, index=blob.index)
    for k in THEMATIC_MUST_ANY:
        mask = mask | blob.str.contains(k, regex=False)
    return mask


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets-dir", default="data/source/datasets")
    p.add_argument("--out-csv", default="data/source/news_datasets.csv")
    p.add_argument("--start-date", default="2015-01-01")
    p.add_argument("--end-date", default="2026-04-30")
    p.add_argument("--min-len", type=int, default=120)
    args = p.parse_args()

    base = Path(args.datasets_dir)
    rows: list[pd.DataFrame] = []

    p1 = base / "data.csv"
    if p1.exists():
        d1 = _read_csv_flexible(p1)
        if {"article_text", "date", "source"}.issubset(set(d1.columns)):
            x = pd.DataFrame()
            x["title"] = ""
            x["text_raw"] = d1["article_text"].fillna("").astype(str)
            x["published_at_utc"] = _to_iso(d1["date"])
            x["source"] = _normalize_source(d1["source"])
            x["url"] = "dataset://data.csv/" + d1.index.astype(str)
            rows.append(x)

    p2 = base / "RussianFinancialNews" / "preview.csv"
    if p2.exists():
        d2 = _read_csv_flexible(p2)
        if {"title", "body", "date", "source"}.issubset(set(d2.columns)):
            x = pd.DataFrame()
            x["title"] = d2["title"].fillna("").astype(str)
            x["text_raw"] = d2["body"].fillna("").astype(str)
            x["published_at_utc"] = _to_iso(d2["date"])
            x["source"] = _normalize_source(d2["source"])
            x["url"] = "dataset://preview.csv/" + d2.index.astype(str)
            rows.append(x)

    if not rows:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(args.out_csv, index=False)
        print(json.dumps({"added_rows": 0, "reason": "no_supported_inputs"}, ensure_ascii=False))
        return

    out = pd.concat(rows, ignore_index=True)
    out = out[out["published_at_utc"].notna()].copy()
    s = pd.to_datetime(args.start_date, utc=True)
    e = pd.to_datetime(args.end_date, utc=True) + pd.Timedelta(days=1)
    ts = pd.to_datetime(out["published_at_utc"], utc=True, errors="coerce")
    out = out[(ts >= s) & (ts < e)].copy()
    out = out[out["text_raw"].str.len() >= int(args.min_len)].copy()
    out = out[_is_thematic(out["title"], out["text_raw"])].copy()

    out["news_id"] = [
        _make_news_id(src, txt[:1000]) for src, txt in zip(out["source"].astype(str), out["text_raw"].astype(str))
    ]
    out["rubric"] = "monetary_policy"
    out["ingestion_ts"] = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    out = out.drop_duplicates(subset=["news_id"], keep="first")
    out = out.drop_duplicates(subset=["text_raw"], keep="first")
    out = out.sort_values("published_at_utc").reset_index(drop=True)

    cols = ["news_id", "source", "url", "published_at_utc", "title", "text_raw", "rubric", "ingestion_ts"]
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out[cols].to_csv(args.out_csv, index=False)
    print(
        json.dumps(
            {
                "added_rows": int(len(out)),
                "sources": out["source"].value_counts().to_dict(),
                "out_csv": args.out_csv,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
