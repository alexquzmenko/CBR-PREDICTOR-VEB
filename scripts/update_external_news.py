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
from typing import Iterable
from urllib.parse import quote_plus, urlparse
from urllib.request import Request, urlopen

import pandas as pd

KEYWORDS = [
    "ключевая ставка",
    "ставка цб",
    "денежно-кредитная политика",
    "банк россии ставка",
    "инфляция цб",
    "решение цб",
    "курс рубля",
    "дкп",
    "жесткая политика",
    "смягчение политики",
]
THEMATIC_MUST_ANY = [
    "ключев",
    "ставк",
    "инфляц",
    "денежно-кредит",
    "совет директор",
    "заявление председателя",
    "набиуллин",
    "рефинансирован",
    "дкп",
    "курс руб",
    "доходност",
    "облигац",
]
SOURCE_SITES = [
    ("rbc.ru", "rbc"),
    ("tass.ru", "tass"),
    ("interfax.ru", "interfax"),
    ("kommersant.ru", "kommersant"),
    ("vedomosti.ru", "vedomosti"),
    ("ria.ru", "ria"),
    ("1prime.ru", "prime"),
    ("banki.ru", "banki"),
    ("finam.ru", "finam"),
    ("investing.com", "investing"),
]


def _fetch(url: str, timeout: int = 25) -> str:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; dataset-bot/1.0)"})
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return raw.decode("utf-8", errors="ignore")


def _html_to_text(html: str) -> str:
    cleaned = re.sub(r"<!--.*?-->", " ", html, flags=re.DOTALL)
    cleaned = re.sub(r"<script.*?>.*?</script>", " ", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<style.*?>.*?</style>", " ", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"</p>|<br\s*/?>|</li>", "\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _extract_article_text(html: str) -> str:
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


def _extract_direct_link(item: ET.Element) -> str:
    desc = item.findtext("description") or ""
    hrefs = re.findall(r'href="([^"]+)"', desc)
    for h in hrefs:
        host = urlparse(h).netloc.lower()
        if "rbc.ru" in host or "tass.ru" in host:
            return h
    link = (item.findtext("link") or "").strip()
    return link


def _resolve_google_link(url: str, timeout: int = 20) -> str:
    """Resolve news.google.com redirect to publisher URL."""
    if "news.google.com" not in urlparse(url).netloc.lower():
        return url
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; dataset-bot/1.0)"})
        with urlopen(req, timeout=timeout) as resp:
            final_url = resp.geturl()
        return final_url or url
    except Exception:
        return url


def _iter_year_ranges(start_date: str, end_date: str) -> list[tuple[str, str]]:
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    cur = datetime(start.year, 1, 1)
    out: list[tuple[str, str]] = []
    while cur <= end.to_pydatetime():
        nxt = datetime(cur.year + 1, 1, 1)
        b = min(nxt, end.to_pydatetime() + pd.Timedelta(days=1))
        out.append((cur.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d")))
        cur = nxt
    return out


def _read_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    csv.field_size_limit(10**8)
    raw = path.read_bytes().replace(b"\x00", b"")
    text = raw.decode("utf-8", errors="ignore")
    return list(csv.DictReader(io.StringIO(text)))


def _write_rows(path: Path, rows: Iterable[dict]) -> None:
    fieldnames = ["news_id", "source", "url", "published_at_utc", "title", "text_raw", "rubric", "ingestion_ts"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _google_rss_url(site: str, keyword: str, after: str, before: str) -> str:
    query = f'site:{site} "{keyword}" after:{after} before:{before}'
    return f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=ru&gl=RU&ceid=RU:ru"


def _is_thematic(title: str, text: str, url: str) -> bool:
    blob = f"{title} {text} {url}".lower()
    return any(k in blob for k in THEMATIC_MUST_ANY)


def update_external_news(
    news_csv: Path, start_date: str, end_date: str, timeout: int, max_items: int, fast_mode: bool
) -> dict:
    existing = _read_existing(news_csv)
    known_urls = {r.get("url", "").strip() for r in existing if r.get("url")}
    known_hash = {
        hashlib.md5((r.get("text_raw", "") or "").strip().encode("utf-8")).hexdigest()
        for r in existing
        if (r.get("text_raw", "") or "").strip()
    }
    rows = list(existing)
    scanned = 0
    added = 0
    now_iso = datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")

    for site, source in SOURCE_SITES:
        for after, before in _iter_year_ranges(start_date, end_date):
            for kw in KEYWORDS:
                rss = _google_rss_url(site=site, keyword=kw, after=after, before=before)
                try:
                    xml = _fetch(rss, timeout=timeout)
                    root = ET.fromstring(xml)
                except Exception:
                    continue
                for item in root.findall(".//item"):
                    if scanned >= max_items:
                        break
                    scanned += 1
                    if fast_mode:
                        url = _extract_direct_link(item).split("#")[0]
                    else:
                        url = _resolve_google_link(_extract_direct_link(item), timeout=timeout).split("#")[0]
                    host = urlparse(url).netloc.lower()
                    if (not fast_mode) and site not in host:
                        continue
                    if url in known_urls:
                        continue
                    title = (item.findtext("title") or "").strip()
                    pub = (item.findtext("pubDate") or "").strip()
                    try:
                        dt = datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo=timezone.utc)
                        pub_iso = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                    except Exception:
                        pub_iso = now_iso
                    if fast_mode:
                        text_raw = _html_to_text(f"{title}. {item.findtext('description') or ''}")
                    else:
                        try:
                            html = _fetch(url, timeout=timeout)
                        except Exception:
                            continue
                        text_raw = _extract_article_text(html)
                    min_len = 120 if fast_mode else 500
                    if len(text_raw) < min_len:
                        continue
                    if not _is_thematic(title, text_raw, url):
                        continue
                    text_hash = hashlib.md5(text_raw.encode("utf-8")).hexdigest()
                    if text_hash in known_hash:
                        continue
                    row = {
                        "news_id": f"{source}_{hashlib.sha1(url.encode('utf-8')).hexdigest()[:12]}",
                        "source": source,
                        "url": url,
                        "published_at_utc": pub_iso,
                        "title": title[:220],
                        "text_raw": text_raw,
                        "rubric": "monetary_policy",
                        "ingestion_ts": now_iso,
                    }
                    rows.append(row)
                    known_urls.add(url)
                    known_hash.add(text_hash)
                    added += 1
                if scanned >= max_items:
                    break
            if scanned >= max_items:
                break

    rows = sorted(rows, key=lambda r: r.get("published_at_utc", ""))
    _write_rows(news_csv, rows)
    return {"news_csv": str(news_csv), "scanned_items": scanned, "added_rows": added, "total_rows": len(rows)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--news-csv", default="data/source/news_external.csv")
    p.add_argument("--start-date", default="2015-01-01")
    p.add_argument("--end-date", default="2026-12-31")
    p.add_argument("--timeout", type=int, default=25)
    p.add_argument("--max-items", type=int, default=12000)
    p.add_argument("--fast-mode", action="store_true")
    args = p.parse_args()
    report = update_external_news(
        news_csv=Path(args.news_csv),
        start_date=args.start_date,
        end_date=args.end_date,
        timeout=args.timeout,
        max_items=args.max_items,
        fast_mode=args.fast_mode,
    )
    print(report)


if __name__ == "__main__":
    main()
