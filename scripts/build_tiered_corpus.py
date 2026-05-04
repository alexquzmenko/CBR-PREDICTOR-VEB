from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _month_iter(start_year: int, end_year: int):
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            yield year, month


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cbr-csv", default="data/source/news_cbr.csv")
    p.add_argument("--external-csv", action="append", default=[])
    p.add_argument("--out-csv", default="data/source/news_tiered.csv")
    p.add_argument("--checkpoint-csv", default="reports/corpus_checkpoint.csv")
    p.add_argument("--skip-log-json", default="reports/corpus_skipped_batches.json")
    p.add_argument("--start-year", type=int, default=2015)
    p.add_argument("--end-year", type=int, default=2026)
    p.add_argument("--queries", nargs="+", default=["ключев", "ставк", "инфляц", "денежно-кредит"])
    p.add_argument("--full-text-min-len", type=int, default=500)
    p.add_argument("--snippet-min-len", type=int, default=120)
    args = p.parse_args()

    cbr = _load_csv(Path(args.cbr_csv))
    external_paths = args.external_csv if args.external_csv else ["data/source/news_rbc.csv", "data/source/news_external.csv"]
    ext_parts = [_load_csv(Path(pth)) for pth in external_paths]
    ext = pd.concat([x for x in ext_parts if not x.empty], ignore_index=True) if ext_parts else pd.DataFrame()
    df = pd.concat([cbr, ext], ignore_index=True)
    if df.empty:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(args.out_csv, index=False)
        print(json.dumps({"saved_rows": 0, "note": "empty inputs"}, ensure_ascii=False))
        return

    df["published_at_utc"] = pd.to_datetime(df["published_at_utc"], errors="coerce", utc=True)
    df = df[df["published_at_utc"].notna()].copy()
    df["text_raw"] = df["text_raw"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["qblob"] = (df["title"] + " " + df["text_raw"]).str.lower()
    df["text_len"] = df["text_raw"].str.len()
    df["text_tier"] = df["text_len"].map(
        lambda x: "full_text" if x >= args.full_text_min_len else ("snippet_only" if x >= args.snippet_min_len else "short")
    )
    df = df[df["text_tier"] != "short"].copy()

    checkpoints: list[dict] = []
    skipped: list[dict] = []
    parts: list[pd.DataFrame] = []
    for year, month in _month_iter(args.start_year, args.end_year):
        monthly = df[(df["published_at_utc"].dt.year == year) & (df["published_at_utc"].dt.month == month)]
        for q in args.queries:
            part = monthly[monthly["qblob"].str.contains(q, na=False)].copy()
            if part.empty:
                skipped.append({"year": year, "month": month, "query": q, "reason": "empty_batch"})
                continue
            part["batch_year"] = year
            part["batch_month"] = month
            part["batch_query"] = q
            checkpoints.append(
                {
                    "year": year,
                    "month": month,
                    "query": q,
                    "rows": int(len(part)),
                    "full_text_rows": int((part["text_tier"] == "full_text").sum()),
                    "snippet_rows": int((part["text_tier"] == "snippet_only").sum()),
                }
            )
            parts.append(part)

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=df.columns)
    if not out.empty:
        out = out.drop_duplicates(subset=["url"], keep="first").sort_values("published_at_utc").reset_index(drop=True)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    pd.DataFrame(checkpoints).to_csv(args.checkpoint_csv, index=False)
    Path(args.skip_log_json).write_text(json.dumps(skipped, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "saved_rows": int(len(out)),
        "full_text_rows": int((out["text_tier"] == "full_text").sum()) if not out.empty else 0,
        "snippet_rows": int((out["text_tier"] == "snippet_only").sum()) if not out.empty else 0,
        "batches_with_data": int(len(checkpoints)),
        "skipped_batches": int(len(skipped)),
        "out_csv": args.out_csv,
        "checkpoint_csv": args.checkpoint_csv,
        "skip_log_json": args.skip_log_json,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
