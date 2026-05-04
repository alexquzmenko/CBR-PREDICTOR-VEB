"""Нарастить news_cbr_work.csv / news_cbr.csv батчами event-id и отчётом прогресса."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_update():
    mod_path = Path(__file__).resolve().parent / "update_cbr_news.py"
    spec = importlib.util.spec_from_file_location("cbr_update", mod_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--news-csv", default="data/source/news_cbr_work.csv")
    p.add_argument("--target", type=int, default=400)
    p.add_argument("--chunk", type=int, default=80, help="размер батча по event id")
    p.add_argument("--timeout", type=int, default=2)
    p.add_argument("--max-links", type=int, default=600)
    p.add_argument(
        "--ranges",
        nargs="*",
        default=[],
        help="пары START END через пробел, например 26800 26999 27000 27199",
    )
    args = p.parse_args()

    m = _load_update()
    news_path = Path(args.news_csv)
    report_path = Path("reports/cbr_update_report.json")

    before = len(pd.read_csv(news_path))
    print(f"[grow] start_rows={before} target={args.target}", flush=True)

    if args.ranges:
        ranges = []
        nums = [int(x) for x in args.ranges]
        for i in range(0, len(nums), 2):
            if i + 1 < len(nums):
                ranges.append((nums[i], nums[i + 1]))
    else:
        # Плотные диапазоны id для press/event (без длинного скана «пустых» id).
        ranges = []
        for a in range(26500, 29200, args.chunk):
            ranges.append((a, a + args.chunk - 1))

    added_total = 0
    for i, (start, end) in enumerate(ranges):
        cur = len(pd.read_csv(news_path))
        if cur >= args.target:
            print(f"[grow] target reached: {cur}", flush=True)
            break
        rep = m.update_news(
            news_csv=news_path,
            report_json=report_path,
            section_urls=m.DEFAULT_SECTION_URLS,
            rss_urls=m.DEFAULT_RSS_URLS,
            max_links=args.max_links,
            timeout=args.timeout,
            include_keypr=False,
            event_id_start=start,
            event_id_end=end,
        )
        cur2 = len(pd.read_csv(news_path))
        added = rep["added_rows"]
        added_total += added
        print(
            f"[grow] chunk {i+1}/{len(ranges)} ids {start}-{end}: +{added} total={cur2}",
            flush=True,
        )

    after = len(pd.read_csv(news_path))
    print(f"[grow] final_rows={after} added_total={after - before}", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
