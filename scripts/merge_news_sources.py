from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def merge_news(cbr_csv: Path, external_csv: Path, out_csv: Path, start_date: str, end_date: str) -> pd.DataFrame:
    cbr = pd.read_csv(cbr_csv) if cbr_csv.exists() else pd.DataFrame()
    ext = pd.read_csv(external_csv) if external_csv.exists() else pd.DataFrame()
    df = pd.concat([cbr, ext], ignore_index=True)
    if df.empty:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        return df
    df["published_at_utc"] = pd.to_datetime(df["published_at_utc"], errors="coerce", utc=True)
    s = pd.to_datetime(start_date, utc=True)
    e = pd.to_datetime(end_date, utc=True) + pd.Timedelta(days=1)
    df = df[(df["published_at_utc"] >= s) & (df["published_at_utc"] < e)].copy()
    df["url"] = df["url"].astype(str).str.strip()
    df["text_raw"] = df["text_raw"].astype(str).str.strip()
    df = df[df["url"] != ""]
    df = df[df["text_raw"].str.len() >= 120]
    df = df.drop_duplicates(subset=["url"], keep="first")
    df = df.sort_values("published_at_utc").reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df["published_at_utc"] = df["published_at_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df.to_csv(out_csv, index=False)
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cbr-csv", default="data/source/news_cbr.csv")
    p.add_argument("--external-csv", default="data/source/news_external.csv")
    p.add_argument("--out-csv", default="data/source/news_all.csv")
    p.add_argument("--start-date", default="2015-01-01")
    p.add_argument("--end-date", default="2026-12-31")
    args = p.parse_args()
    out = merge_news(
        cbr_csv=Path(args.cbr_csv),
        external_csv=Path(args.external_csv),
        out_csv=Path(args.out_csv),
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(f"saved {len(out)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
