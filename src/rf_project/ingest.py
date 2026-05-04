from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .io_utils import read_table, save_table


@dataclass
class IngestOutputs:
    news_path: Path
    macro_path: Path
    rates_path: Path


def _demo_news(n: int, start_date: str, end_date: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(rng.choice(pd.date_range(start_date, end_date, freq="D"), size=n))
    sources = rng.choice(["rbc", "tass", "banki", "finam", "investing"], size=n)
    sentiments = rng.choice(["negative", "neutral", "positive"], size=n, p=[0.35, 0.4, 0.25])
    events = rng.choice(
        [
            "инфляция растет",
            "курс рубля укрепляется",
            "рынок ожидает решение по ставке",
            "комментарии ЦБ о рисках",
            "санкционные новости усиливают неопределенность",
        ],
        size=n,
    )
    rows = []
    for i in range(n):
        txt = (
            f"Новость {i}. {events[i]}. Источник сообщает {sentiments[i]} ожидания. "
            "Банк России, инфляция и рубль обсуждаются в контексте ключевой ставки."
        )
        rows.append(
            {
                "news_id": f"news_{i}",
                "source": sources[i],
                "url": f"https://example.org/{sources[i]}/{i}",
                "published_at_utc": dates[i],
                "title": f"Экономическая новость {i}",
                "text_raw": txt,
                "rubric": "economy",
                "ingestion_ts": pd.Timestamp.utcnow(),
            }
        )
    return pd.DataFrame(rows)


def _demo_macro(start_date: str, end_date: str, seed: int) -> pd.DataFrame:
    dates = pd.date_range(start_date, end_date, freq="D")
    rng = np.random.default_rng(seed + 1)
    return pd.DataFrame(
        {
            "feature_date": dates,
            "cpi_yoy": 5 + np.clip(np.cumsum(rng.normal(0, 0.03, len(dates))), -2, 5),
            "usd_rub": 85 + np.clip(np.cumsum(rng.normal(0, 0.4, len(dates))), -10, 20),
            "brent": 75 + np.clip(np.cumsum(rng.normal(0, 0.2, len(dates))), -15, 15),
            "rvi": 25 + np.abs(rng.normal(0, 1.5, len(dates))),
            "pmi": 50 + rng.normal(0, 1.2, len(dates)),
        }
    )


def _demo_rates(start_date: str, end_date: str, seed: int) -> pd.DataFrame:
    meetings = pd.date_range(start_date, end_date, freq="6W-FRI")
    rng = np.random.default_rng(seed + 2)
    current = 8.5
    out = []
    for d in meetings:
        move = rng.choice([-1, 0, 1], p=[0.2, 0.5, 0.3])
        delta = move * rng.choice([25, 50], p=[0.75, 0.25])
        next_rate = max(1.0, current + delta / 100.0)
        cls = "up" if delta > 0 else ("down" if delta < 0 else "hold")
        out.append(
            {
                "meeting_date": d,
                "rate_before": current,
                "rate_after": next_rate,
                "delta_bp": int(delta),
                "target_class": cls,
            }
        )
        current = next_rate
    return pd.DataFrame(out)


def _macro_daily_filled(macro: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Полный календарь дней; пропуски заполняются ffill/bfill (ранние даты — от первого известного ряда)."""
    macro = macro.copy()
    macro["feature_date"] = pd.to_datetime(macro["feature_date"]).dt.normalize()
    macro = macro.sort_values("feature_date").drop_duplicates("feature_date", keep="last")
    dr = pd.DataFrame({"feature_date": pd.date_range(start_date, end_date, freq="D").normalize()})
    out = dr.merge(macro, on="feature_date", how="left").sort_values("feature_date")
    num = [c for c in out.columns if c != "feature_date"]
    out[num] = out[num].ffill().bfill()
    return out


def run_ingest(cfg: dict) -> IngestOutputs:
    raw_dir = Path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    ds = cfg["dataset"]
    src = cfg["sources"]

    if src.get("use_demo_generator", True):
        demo_news_count = int(src.get("demo_news_count", 2000))
        news = _demo_news(demo_news_count, ds["start_date"], ds["end_date"], ds["random_seed"])
        macro = _demo_macro(ds["start_date"], ds["end_date"], ds["random_seed"])
        rates = _demo_rates(ds["start_date"], ds["end_date"], ds["random_seed"])
    else:
        news = read_table(src["news_csv"])
        macro = read_table(src["macro_csv"])
        rates = read_table(src["rates_csv"])
        macro = _macro_daily_filled(macro, ds["start_date"], ds["end_date"])

    news_path = raw_dir / "news_raw.parquet"
    macro_path = raw_dir / "macro_raw.parquet"
    rates_path = raw_dir / "rates_raw.parquet"
    save_table(news, news_path)
    save_table(macro, macro_path)
    save_table(rates, rates_path)
    return IngestOutputs(news_path=news_path, macro_path=macro_path, rates_path=rates_path)
