from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from webdriver_manager.chrome import ChromeDriverManager
except Exception:  # pragma: no cover - optional runtime dependency
    webdriver = None


KEYWORDS = [
    "ключевая ставка",
    "ставка цб",
    "банк россии",
    "денежно-кредитная политика",
    "инфляция",
]


def _fetch(url: str, timeout: int = 25) -> str:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; dataset-bot/1.0)"})
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    for enc in ("utf-8", "cp1251", "windows-1251", "latin-1"):
        try:
            return raw.decode(enc, errors="ignore")
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
            if len(txt) > 300:
                return txt
    return _html_to_text(html)


def _read_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    csv.field_size_limit(10**8)
    raw = path.read_bytes().replace(b"\x00", b"")
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8", errors="ignore"))))


def _write_rows(path: Path, rows: list[dict]) -> None:
    fields = ["news_id", "source", "url", "published_at_utc", "title", "text_raw", "rubric", "ingestion_ts"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _is_thematic(title: str, text: str, url: str) -> bool:
    blob = f"{title} {text} {url}".lower()
    return any(k in blob for k in ["ключев", "ставк", "цб", "банк россии", "инфляц", "денежно-кредит"])


def _collect_links_selenium(keyword: str, start_date: str, end_date: str, max_links: int) -> list[str]:
    if webdriver is None:
        return []
    q = quote_plus(keyword)
    url = f"https://www.rbc.ru/search/?query={q}&project=rbcnews&from={start_date}&to={end_date}"
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
    links: list[str] = []
    try:
        driver.get(url)
        for _ in range(15):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.0)
        anchors = driver.find_elements(By.TAG_NAME, "a")
        for a in anchors:
            href = (a.get_attribute("href") or "").strip()
            if "rbc.ru" not in href:
                continue
            if "/rbcfreenews/" in href or "/economics/" in href or "/business/" in href or "/finances/" in href:
                links.append(href.split("#")[0])
    except Exception:
        return []
    finally:
        driver.quit()
    dedup = []
    seen = set()
    for u in links:
        if u not in seen:
            seen.add(u)
            dedup.append(u)
    return dedup[:max_links]


def _collect_rss_items(max_links: int) -> list[dict]:
    feeds = [
        "https://rssexport.rbc.ru/rbcnews/news/30/full.rss",
        "https://rssexport.rbc.ru/rbcnews/news/100/full.rss",
    ]
    items_out: list[dict] = []
    for f in feeds:
        try:
            xml = _fetch(f, timeout=20)
            root = ET.fromstring(xml)
            for item in root.findall(".//item"):
                lk = (item.findtext("link") or "").strip()
                if not lk:
                    continue
                items_out.append(
                    {
                        "url": lk.split("#")[0],
                        "title": item.findtext("title") or "",
                        "description": item.findtext("description") or "",
                        "pubDate": item.findtext("pubDate") or "",
                    }
                )
        except Exception:
            continue
    out, seen = [], set()
    for it in items_out:
        u = it["url"]
        if u not in seen:
            seen.add(u)
            out.append(it)
    return out[:max_links]


def fetch_rbc(out_csv: Path, start_date: str, end_date: str, max_links: int, use_selenium: bool) -> dict:
    existing = _read_existing(out_csv)
    known_urls = {r.get("url", "").strip() for r in existing if r.get("url")}
    rows = list(existing)
    now_iso = datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")

    links: list[str] = []
    rss_items: list[dict] = []
    if use_selenium:
        for kw in KEYWORDS:
            links.extend(_collect_links_selenium(kw, start_date, end_date, max_links=max_links))
    if not links:
        rss_items = _collect_rss_items(max_links=max_links)
        links = [x["url"] for x in rss_items]

    scanned = 0
    added = 0
    for url in links:
        if url in known_urls:
            continue
        scanned += 1
        rss_item = next((x for x in rss_items if x["url"] == url), None)
        title = _html_to_text((rss_item["title"] if rss_item else ""))[:220]
        text_raw = ""
        html = ""
        try:
            html = _fetch(url, timeout=20)
            if html:
                title_m = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.DOTALL | re.IGNORECASE)
                if title_m:
                    title = _html_to_text(title_m.group(1))[:220]
                text_raw = _extract_text(html)
        except Exception:
            pass
        if len(text_raw) < 120 and rss_item is not None:
            text_raw = _html_to_text(f"{rss_item['title']} {rss_item['description']}")
        if len(text_raw) < 120 or not _is_thematic(title, text_raw, url):
            continue
        dt_m = re.search(r'"datePublished"\s*:\s*"([^"]+)"', html)
        pub = dt_m.group(1) if dt_m else (rss_item["pubDate"] if rss_item else now_iso)
        try:
            pub_ts = datetime.fromisoformat(pub.replace("Z", "+00:00"))
            pub_iso = pub_ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            pub_iso = now_iso
        rows.append(
            {
                "news_id": f"rbc_{hashlib.sha1(url.encode('utf-8')).hexdigest()[:12]}",
                "source": "rbc",
                "url": url,
                "published_at_utc": pub_iso,
                "title": title,
                "text_raw": text_raw,
                "rubric": "monetary_policy",
                "ingestion_ts": now_iso,
            }
        )
        known_urls.add(url)
        added += 1
    rows = sorted(rows, key=lambda r: r.get("published_at_utc", ""))
    _write_rows(out_csv, rows)
    return {"news_csv": str(out_csv), "scanned_links": scanned, "added_rows": added, "total_rows": len(rows)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--news-csv", default="data/source/news_rbc.csv")
    p.add_argument("--start-date", default="2015-01-01")
    p.add_argument("--end-date", default="2026-12-31")
    p.add_argument("--max-links", type=int, default=2000)
    p.add_argument("--no-selenium", action="store_true")
    args = p.parse_args()
    report = fetch_rbc(
        out_csv=Path(args.news_csv),
        start_date=args.start_date,
        end_date=args.end_date,
        max_links=args.max_links,
        use_selenium=not args.no_selenium,
    )
    print(report)


if __name__ == "__main__":
    main()
