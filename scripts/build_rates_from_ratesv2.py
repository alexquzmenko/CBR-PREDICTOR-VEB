from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load_daily_rates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df = df.rename(columns={"Дата": "date", "Ставка": "rate"})
    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y", errors="coerce")
    df["rate"] = (
        df["rate"].astype(str).str.replace(",", ".", regex=False).str.replace(" ", "", regex=False)
    )
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    df = df.dropna(subset=["date", "rate"]).sort_values("date").reset_index(drop=True)
    return df


def _load_existing_meetings(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["meeting_date", "rate_before", "rate_after", "delta_bp", "target_class"])
    df = pd.read_csv(path)
    df["meeting_date"] = pd.to_datetime(df["meeting_date"], errors="coerce")
    return df.dropna(subset=["meeting_date"]).sort_values("meeting_date").reset_index(drop=True)


def build_rates(rates_v2: Path, old_rates: Path, out_csv: Path, start_date: str) -> pd.DataFrame:
    daily = _load_daily_rates(rates_v2)
    daily = daily[daily["date"] >= pd.to_datetime(start_date)].copy()

    changes = daily[daily["rate"] != daily["rate"].shift(1)].copy()
    changes["rate_before"] = changes["rate"].shift(1)
    changes["rate_after"] = changes["rate"]
    changes = changes.dropna(subset=["rate_before"]).copy()
    changes["delta_bp"] = ((changes["rate_after"] - changes["rate_before"]) * 100).round().astype(int)
    changes["target_class"] = changes["delta_bp"].map(lambda x: "up" if x > 0 else ("down" if x < 0 else "hold"))
    inferred = changes.rename(columns={"date": "meeting_date"})[
        ["meeting_date", "rate_before", "rate_after", "delta_bp", "target_class"]
    ]

    existing = _load_existing_meetings(old_rates)
    if not existing.empty:
        existing = existing[existing["meeting_date"] >= pd.to_datetime(start_date)].copy()
        hold_rows = existing[existing["target_class"].astype(str) == "hold"].copy()
        # Keep explicit hold meetings (cannot be inferred from daily levels).
        merged = pd.concat([inferred, hold_rows], ignore_index=True)
    else:
        merged = inferred.copy()

    merged = merged.drop_duplicates(subset=["meeting_date"], keep="first")
    merged = merged.sort_values("meeting_date").reset_index(drop=True)
    merged["meeting_date"] = merged["meeting_date"].dt.strftime("%Y-%m-%d")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    return merged


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rates-v2", default="data/source/ratesv2.csv")
    p.add_argument("--old-rates", default="data/source/rates.csv")
    p.add_argument("--out-csv", default="data/source/rates.csv")
    p.add_argument("--start-date", default="2015-01-01")
    args = p.parse_args()

    out = build_rates(
        rates_v2=Path(args.rates_v2),
        old_rates=Path(args.old_rates),
        out_csv=Path(args.out_csv),
        start_date=args.start_date,
    )
    print(f"saved {len(out)} meetings to {args.out_csv}")


if __name__ == "__main__":
    main()
