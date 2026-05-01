from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import io
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


DEFAULT_SECTION_URLS = [
    "https://www.cbr.ru/press/pr/",
    "https://www.cbr.ru/press/event/",
    # Новости по денежно-кредитной политике (приоритетный раздел).
    "https://www.cbr.ru/dkp/news/",
    "https://www.cbr.ru/dkp/",
]
DEFAULT_RSS_URLS = [
    "https://www.cbr.ru/rss/RssNews",
]


def _fetch(url: str, timeout: int = 25) -> str:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; dataset-bot/1.0)"})
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return raw.decode("utf-8", errors="ignore")


def _extract_links(html: str, base_url: str) -> list[str]:
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    out: list[str] = []
    for href in hrefs:
        abs_url = urljoin(base_url, href)
        parsed = urlparse(abs_url)
        if parsed.scheme not in {"http", "https"}:
            continue
        if parsed.netloc not in {"www.cbr.ru", "cbr.ru"}:
            continue
        out.append(abs_url.split("#")[0])
    return out


def _extract_links_from_rss(rss_xml: str) -> list[str]:
    links: list[str] = []
    try:
        root = ET.fromstring(rss_xml)
    except ET.ParseError:
        return links
    for item in root.findall(".//item"):
        link = (item.findtext("link") or "").strip()
        if link:
            links.append(link)
    return links


def _is_candidate_article(url: str) -> bool:
    """Страницы-публикации: press, документы, материалы /news/, разделы /dkp/ (кроме календаря)."""
    p = urlparse(url)
    path = p.path.lower()
    query = (p.query or "").strip()

    if "/press/" in path or "/content/document/" in path:
        return True

    # Новости Банка (не только главная лента без параметров).
    if path.startswith("/news"):
        if path in ("/news", "/news/") and not query:
            return False
        return True

    # ДКП: материалы подразделов, не корневая страница без параметров.
    if path.startswith("/dkp"):
        if "cal_mp" in path:
            return False
        if path in ("/dkp", "/dkp/") and not query:
            return False
        return True

    return False


def _is_allowed_url(url: str, include_keypr: bool) -> bool:
    p = urlparse(url)
    path = p.path.lower().rstrip("/")
    blocked_contains = [
        "/dkp/cal_mp",
        "/analytics/na_vr",
        "/content/document/file/",
    ]
    if any(x in path for x in blocked_contains):
        return False
    if path.endswith(".pdf"):
        return False
    if path == "/press/pr":
        return False
    if path == "/press/keypr" and not include_keypr:
        return False
    return True


def _html_to_text(html: str) -> str:
    # Remove scripts/styles/comments.
    cleaned = re.sub(r"<!--.*?-->", " ", html, flags=re.DOTALL)
    cleaned = re.sub(r"<script.*?>.*?</script>", " ", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<style.*?>.*?</style>", " ", cleaned, flags=re.DOTALL | re.IGNORECASE)
    # Convert paragraphs/line breaks into separators before stripping tags.
    cleaned = re.sub(r"</p>|<br\\s*/?>|</li>", "\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _extract_title(html: str) -> str:
    h1 = re.search(r"<h1[^>]*>(.*?)</h1>", html, flags=re.DOTALL | re.IGNORECASE)
    if h1:
        return _html_to_text(h1.group(1))[:220]
    og = re.search(
        r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\'](.*?)["\']',
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if og:
        return _html_to_text(og.group(1))[:220]
    tt = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.DOTALL | re.IGNORECASE)
    if tt:
        return _html_to_text(tt.group(1))[:220]
    return "CBR publication"


def _extract_datetime_utc(html: str, fallback_url: str) -> str:
    # Common CBR-style date/time in page text: DD.MM.YYYY HH:MM:SS
    m = re.search(r"(\d{2}\.\d{2}\.\d{4})\s+(\d{2}:\d{2}:\d{2})", html)
    if m:
        dt = datetime.strptime(f"{m.group(1)} {m.group(2)}", "%d.%m.%Y %H:%M:%S")
        # MSK -> UTC
        dt_utc = dt.replace(tzinfo=timezone.utc).timestamp() - 3 * 3600
        return datetime.fromtimestamp(dt_utc, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Russian textual date: "24 апреля 2026 года" (time fixed to 13:30 MSK for policy publications).
    month_map = {
        "января": 1,
        "февраля": 2,
        "марта": 3,
        "апреля": 4,
        "мая": 5,
        "июня": 6,
        "июля": 7,
        "августа": 8,
        "сентября": 9,
        "октября": 10,
        "ноября": 11,
        "декабря": 12,
    }
    m_ru = re.search(
        r"\b(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+(\d{4})\s+года\b",
        html.lower(),
    )
    if m_ru:
        day = int(m_ru.group(1))
        month = month_map[m_ru.group(2)]
        year = int(m_ru.group(3))
        return f"{year:04d}-{month:02d}-{day:02d}T10:30:00Z"

    # Fallback: date in URL path if present.
    m2 = re.search(r"/(\d{4})-(\d{2})-(\d{2})/", fallback_url)
    if m2:
        return f"{m2.group(1)}-{m2.group(2)}-{m2.group(3)}T10:30:00Z"
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _extract_article_text(html: str) -> str:
    # Try main/article block first.
    for pattern in [
        r"<article[^>]*>(.*?)</article>",
        r'<div[^>]+class=["\'][^"\']*content[^"\']*["\'][^>]*>(.*?)</div>',
        r'<main[^>]*>(.*?)</main>',
    ]:
        m = re.search(pattern, html, flags=re.DOTALL | re.IGNORECASE)
        if m:
            txt = _html_to_text(m.group(1))
            if len(txt) > 400:
                return txt
    return _html_to_text(html)


def _is_thematic(title: str, text: str, url: str) -> bool:
    blob = f"{title} {text} {url}".lower()
    must_any = [
        "ключев",
        "ставк",
        "инфляц",
        "денежно-кредит",
        "совет директор",
        "заявление председателя",
        "набиуллин",
        "рефинансирован",
    ]
    return any(k in blob for k in must_any)


def _build_news_id(url: str) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return f"cbr_web_{digest}"


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


def update_news(
    news_csv: Path,
    report_json: Path,
    section_urls: list[str],
    rss_urls: list[str],
    max_links: int,
    timeout: int,
    include_keypr: bool,
    event_id_start: int | None,
    event_id_end: int | None,
) -> dict:
    existing_raw = _read_existing(news_csv)
    existing = []
    removed_existing_rows = 0
    for r in existing_raw:
        url = (r.get("url") or "").strip()
        if url and not _is_allowed_url(url, include_keypr=include_keypr):
            removed_existing_rows += 1
            continue
        existing.append(r)
    known_urls = {r.get("url", "").strip() for r in existing if r.get("url", "").strip()}
    known_hash = {
        hashlib.md5((r.get("text_raw", "") or "").strip().encode("utf-8")).hexdigest()
        for r in existing
        if (r.get("text_raw", "") or "").strip()
    }

    seed_links: list[str] = []
    # 1) RSS is the primary source (reliable links for JS-rendered pages).
    for rss_url in rss_urls:
        try:
            rss_xml = _fetch(rss_url, timeout=timeout)
            links = _extract_links_from_rss(rss_xml)
            for link in links:
                if _is_candidate_article(link) and _is_allowed_url(link, include_keypr=include_keypr):
                    seed_links.append(link)
        except Exception:
            continue

    # 2) HTML section scan as fallback.
    for section_url in section_urls:
        try:
            html = _fetch(section_url, timeout=timeout)
            links = _extract_links(html, section_url)
            for link in links:
                if _is_candidate_article(link) and _is_allowed_url(link, include_keypr=include_keypr):
                    seed_links.append(link)
        except Exception:
            continue

    event_links: list[str] = []
    # 3) Optional direct scan for /press/event/?id=... pages (не режем вместе с seed по max_links).
    if event_id_start is not None and event_id_end is not None and event_id_end >= event_id_start:
        for event_id in range(event_id_start, event_id_end + 1):
            u = f"https://www.cbr.ru/press/event/?id={event_id}"
            if _is_allowed_url(u, include_keypr=include_keypr):
                event_links.append(u)

    def _dedup(seq: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for link in seq:
            if link not in seen:
                seen.add(link)
                out.append(link)
        return out

    seed_dedup = _dedup(seed_links)[:max_links]
    event_dedup = _dedup(event_links)
    dedup_links = seed_dedup + event_dedup

    added = 0
    scanned = 0
    new_rows: list[dict] = []
    now_iso = datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")

    for url in dedup_links:
        scanned += 1
        if url in known_urls:
            continue
        try:
            html = _fetch(url, timeout=timeout)
        except Exception:
            continue

        title = _extract_title(html)
        text_raw = _extract_article_text(html)
        if len(text_raw) < 300:
            continue
        if not _is_thematic(title, text_raw, url):
            continue
        text_hash = hashlib.md5(text_raw.encode("utf-8")).hexdigest()
        if text_hash in known_hash:
            continue

        row = {
            "news_id": _build_news_id(url),
            "source": "cbr",
            "url": url,
            "published_at_utc": _extract_datetime_utc(html, url),
            "title": title,
            "text_raw": text_raw,
            "rubric": "monetary_policy",
            "ingestion_ts": now_iso,
        }
        new_rows.append(row)
        known_urls.add(url)
        known_hash.add(text_hash)
        added += 1

    merged = existing + new_rows
    _write_rows(news_csv, merged)

    report = {
        "news_csv": str(news_csv),
        "scanned_links": scanned,
        "added_rows": added,
        "total_rows": len(merged),
        "updated_at_utc": now_iso,
        "removed_existing_rows": removed_existing_rows,
        "rss_urls": rss_urls,
        "event_id_start": event_id_start,
        "event_id_end": event_id_end,
        "section_urls": section_urls,
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--news-csv", default="data/source/news_cbr.csv")
    parser.add_argument("--report-json", default="reports/cbr_update_report.json")
    parser.add_argument("--max-links", type=int, default=250)
    parser.add_argument("--timeout", type=int, default=25)
    parser.add_argument("--section-url", action="append", default=[])
    parser.add_argument("--rss-url", action="append", default=[])
    parser.add_argument("--include-keypr", action="store_true")
    parser.add_argument("--event-id-start", type=int, default=None)
    parser.add_argument("--event-id-end", type=int, default=None)
    args = parser.parse_args()

    section_urls = args.section_url if args.section_url else DEFAULT_SECTION_URLS
    rss_urls = args.rss_url if args.rss_url else DEFAULT_RSS_URLS
    report = update_news(
        news_csv=Path(args.news_csv),
        report_json=Path(args.report_json),
        section_urls=section_urls,
        rss_urls=rss_urls,
        max_links=args.max_links,
        timeout=args.timeout,
        include_keypr=args.include_keypr,
        event_id_start=args.event_id_start,
        event_id_end=args.event_id_end,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
