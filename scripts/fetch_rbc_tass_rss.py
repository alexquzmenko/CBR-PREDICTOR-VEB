from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


RSS_URLS = [
    ("rbc", "https://rssexport.rbc.ru/rbcnews/news/30/full.rss"),
    ("rbc", "https://rssexport.rbc.ru/rbcnews/news/100/full.rss"),
    ("tass", "https://tass.ru/rss/v2.xml"),
]
KEYWORDS = [
    "ключев",
    "ставк",
    "денежно-кредит",
    "инфляц",
    "банк россии",
    "цб",
    "эконом",
    "финанс",
    "рубл",
]


def _fetch(url: str, timeout: int = 20) -> str:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; dataset-bot/1.0)"})
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        enc = None
        try:
            enc = resp.headers.get_content_charset()
        except Exception:
            enc = None
    for candidate in [enc, "utf-8", "cp1251", "windows-1251", "latin-1"]:
        if not candidate:
            continue
        try:
            return raw.decode(candidate, errors="ignore")
        except Exception:
            continue
    return raw.decode("utf-8", errors="ignore")


def _html_to_text(html: str) -> str:
    cleaned = re.sub(r"<!--.*?-->", " ", html, flags=re.DOTALL)
    cleaned = re.sub(r"<script.*?>.*?</script>", " ", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<style.*?>.*?</style>", " ", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"</p>|<br\s*/?>|</li>", "\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = unescape(cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _extract_text(html: str) -> str:
    for pattern in [
        r"<article[^>]*>(.*?)</article>",
        r'<div[^>]+class=["\'][^"\']*(article|content|text)[^"\']*["\'][^>]*>(.*?)</div>',
        r"<main[^>]*>(.*?)</main>",
    ]:
        m = re.search(pattern, html, flags=re.DOTALL | re.IGNORECASE)
        if m:
            txt = _html_to_text(m.group(1))
            if len(txt) > 400:
                return txt
    return _html_to_text(html)


def _is_thematic(title: str, text: str, url: str) -> bool:
    blob = f"{title} {text} {url}".lower()
    if any(k in blob for k in KEYWORDS):
        return True
    u = url.lower()
    if "tass.ru/ekonomika" in u or "tass.ru/ekonomika-i-biznes" in u or "tass.ru/biznes" in u:
        return True
    if "rbc.ru/economics" in u or "rbc.ru/finances" in u or "rbc.ru/business" in u:
        return True
    return False


def _read_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    csv.field_size_limit(10**8)
    raw = path.read_bytes().replace(b"\x00", b"")
    text = raw.decode("utf-8", errors="ignore")
    return list(csv.DictReader(io.StringIO(text)))


def _write_rows(path: Path, rows: list[dict]) -> None:
    fields = ["news_id", "source", "url", "published_at_utc", "title", "text_raw", "rubric", "ingestion_ts"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--news-csv", default="data/source/news_external.csv")
    p.add_argument("--start-date", default="2015-01-01")
    p.add_argument("--end-date", default="2026-12-31")
    p.add_argument("--timeout", type=int, default=20)
    args = p.parse_args()

    start = pd.to_datetime(args.start_date, utc=True)
    end = pd.to_datetime(args.end_date, utc=True) + pd.Timedelta(days=1)
    out_path = Path(args.news_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_existing(out_path)
    known = {r.get("url", "").strip() for r in existing if r.get("url")}
    rows = list(existing)
    added = 0
    scanned = 0
    now_iso = datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")

    for source, rss_url in RSS_URLS:
        try:
            xml = _fetch(rss_url, timeout=args.timeout)
            root = ET.fromstring(xml)
        except Exception:
            continue
        for item in root.findall(".//item"):
            url = (item.findtext("link") or "").strip().split("#")[0]
            if not url or url in known:
                continue
            host = urlparse(url).netloc.lower()
            if source == "rbc" and "rbc.ru" not in host:
                continue
            if source == "tass" and "tass.ru" not in host:
                continue
            scanned += 1
            title = _html_to_text(item.findtext("title") or "")[:220]
            try:
                pub = pd.to_datetime(item.findtext("pubDate"), utc=True)
            except Exception:
                pub = pd.Timestamp.now(tz="UTC")
            if pub < start or pub >= end:
                continue
            text_raw = _html_to_text(f"{title}. {item.findtext('description') or ''}")
            if len(text_raw) < 120 or not _is_thematic(title, text_raw, url):
                continue
            rows.append(
                {
                    "news_id": f"{source}_{hashlib.sha1(url.encode('utf-8')).hexdigest()[:12]}",
                    "source": source,
                    "url": url,
                    "published_at_utc": pub.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "title": title,
                    "text_raw": text_raw,
                    "rubric": "monetary_policy",
                    "ingestion_ts": now_iso,
                }
            )
            known.add(url)
            added += 1

    rows = sorted(rows, key=lambda r: r.get("published_at_utc", ""))
    _write_rows(out_path, rows)
    print({"news_csv": str(out_path), "scanned": scanned, "added": added, "total_rows": len(rows)})


if __name__ == "__main__":
    import pandas as pd

    main()
