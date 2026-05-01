from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def _load_market_series(path: Path, value_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["feature_date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    df[value_col] = pd.to_numeric(df["Price"], errors="coerce")
    return df[["feature_date", value_col]].dropna(subset=["feature_date"]).sort_values("feature_date")


def _load_fred_brent(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["feature_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    # FRED may store missing values as "." strings.
    df["brent"] = pd.to_numeric(df["DCOILBRENTEU"], errors="coerce")
    return df[["feature_date", "brent"]].dropna(subset=["feature_date"]).sort_values("feature_date")


def _load_ipi_activity_proxy(path: Path) -> pd.DataFrame:
    # Rosstat file is semicolon-delimited with metadata/header rows.
    raw = pd.read_csv(path, sep=";", header=None, dtype=str, encoding="utf-8")
    row = raw[raw.iloc[:, 0].fillna("").str.startswith("Промышленное производство", na=False)]
    if row.empty:
        return pd.DataFrame(columns=["feature_date", "activity_proxy"])

    values = row.iloc[0].tolist()
    header = raw.iloc[3].tolist() if len(raw) > 3 else []
    pairs = []
    max_len = min(len(header), len(values))
    for i in range(max_len):
        h = str(header[i]) if header[i] is not None else ""
        v = str(values[i]) if values[i] is not None else ""
        m = re.search(r"(\d{4})", h)
        if not m:
            continue
        year = int(m.group(1))
        if year < 2021 or year > 2030:
            continue
        val = pd.to_numeric(v.replace(",", "."), errors="coerce")
        if pd.isna(val):
            continue
        pairs.append((year, float(val)))

    if not pairs:
        return pd.DataFrame(columns=["feature_date", "activity_proxy"])

    out = pd.DataFrame(pairs, columns=["year", "activity_proxy"]).drop_duplicates("year").sort_values("year")
    out["feature_date"] = pd.to_datetime(out["year"].astype(str) + "-01-01")
    return out[["feature_date", "activity_proxy"]]


def build_macro(
    cpi_releases_path: Path,
    usdrub_path: Path,
    rvi_path: Path,
    brent_path: Path,
    ipi_path: Path,
    output_path: Path,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    cpi = pd.read_csv(cpi_releases_path)
    cpi["release_date"] = pd.to_datetime(cpi["release_date"], errors="coerce")
    cpi["cpi_yoy"] = pd.to_numeric(cpi["cpi_yoy"], errors="coerce")
    cpi = cpi.dropna(subset=["release_date", "cpi_yoy"]).sort_values("release_date")

    dates = pd.DataFrame({"feature_date": pd.date_range(start_date, end_date, freq="D")})

    # Anti-leakage: each day gets the latest CPI already published on or before that day.
    out = pd.merge_asof(
        dates.sort_values("feature_date"),
        cpi.rename(columns={"release_date": "feature_date"})[["feature_date", "cpi_yoy"]].sort_values("feature_date"),
        on="feature_date",
        direction="backward",
    )

    usdrub = _load_market_series(usdrub_path, "usd_rub")
    rvi = _load_market_series(rvi_path, "rvi")
    brent = _load_fred_brent(brent_path)
    ipi = _load_ipi_activity_proxy(ipi_path)
    out = out.merge(usdrub, on="feature_date", how="left")
    out = out.merge(rvi, on="feature_date", how="left")
    out = out.merge(brent, on="feature_date", how="left")
    out = out.merge(ipi, on="feature_date", how="left")

    # Preserve daily continuity for market rows even over missing trading dates.
    out["usd_rub"] = out["usd_rub"].ffill()
    out["rvi"] = out["rvi"].ffill()
    out["brent"] = out["brent"].ffill()
    # Substitute PMI with industrial production activity proxy (year-level, forward-filled).
    out["pmi"] = out["activity_proxy"].ffill()
    out = out[["feature_date", "cpi_yoy", "usd_rub", "brent", "rvi", "pmi"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpi", default="data/source/macro/cpi_releases.csv")
    parser.add_argument("--usdrub", default="data/source/macro/usdrub.csv")
    parser.add_argument("--rvi", default="data/source/macro/rvi.csv")
    parser.add_argument("--brent", default="data/source/macro/DCOILBRENTEU.csv")
    parser.add_argument("--ipi", default="data/source/macro/ind_god_2015-2025.csv")
    parser.add_argument("--output", default="data/source/macro.csv")
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2026-04-24")
    args = parser.parse_args()

    df = build_macro(
        cpi_releases_path=Path(args.cpi),
        usdrub_path=Path(args.usdrub),
        rvi_path=Path(args.rvi),
        brent_path=Path(args.brent),
        ipi_path=Path(args.ipi),
        output_path=Path(args.output),
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
