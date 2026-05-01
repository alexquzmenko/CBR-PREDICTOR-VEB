from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from urllib.parse import quote_plus, urlparse
from urllib.request import Request, urlopen

import pandas as pd

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
KEYWORDS = [
    "inflation",
    "key rate",
    "central bank russia",
    "monetary policy",
    "ruble",
    "banking sector",
]
THEMATIC_MUST_ANY = [
    "ключев",
    "ставк",
    "инфляц",
    "денежно-кредит",
    "банк россии",
    "цб",
]
SOURCE_MAP = {
    "rbc.ru": "rbc",
    "tass.ru": "tass",
    "interfax.ru": "interfax",
    "kommersant.ru": "kommersant",
    "vedomosti.ru": "vedomosti",
    "ria.ru": "ria",
    "1prime.ru": "prime",
}


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
        r'<div[^>]+class=["\'][^"\']*(article|content|text|body)[^"\']*["\'][^>]*>(.*?)</div>',
        r"<main[^>]*>(.*?)</main>",
    ]:
        m = re.search(pattern, html, flags=re.DOTALL | re.IGNORECASE)
        if m:
            txt = _html_to_text(m.group(1))
            if len(txt) > 400:
                return txt
    return _html_to_text(html)


def _source_from_url(url: str) -> str:
    host = urlparse(url).netloc.lower()
    for d, s in SOURCE_MAP.items():
        if d in host:
            return s
    return "external"


def _is_thematic(title: str, text: str, url: str) -> bool:
    blob = f"{title} {text} {url}".lower()
    return any(k in blob for k in THEMATIC_MUST_ANY)


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


def _month_ranges(start_date: str, end_date: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    s = pd.to_datetime(start_date, utc=True)
    e = pd.to_datetime(end_date, utc=True)
    out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = pd.Timestamp(year=s.year, month=s.month, day=1, tz="UTC")
    while cur <= e:
        nxt = (cur + pd.offsets.MonthBegin(1)).tz_convert("UTC")
        left = max(cur, s)
        right = min(nxt - pd.Timedelta(seconds=1), e)
        if left <= right:
            out.append((left, right))
        cur = nxt
    return out


def _gdelt_query(keyword: str, domain: str) -> str:
    # GDELT DOC query language
    return f'"{keyword}" domain:{domain}'


def _to_gdelt_dt(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y%m%d%H%M%S")


def _extract_candidate(
    article: dict,
    timeout: int,
    min_text_len: int,
) -> dict | None:
    link = (article.get("url") or "").strip().split("#")[0]
    title = str(article.get("title") or "").strip()[:220]
    if not link:
        return None
    try:
        html = _fetch(link, timeout=timeout)
    except Exception:
        return None
    text_raw = _extract_text(html)
    if len(text_raw) < min_text_len:
        return None
    if not _is_thematic(title, text_raw, link):
        return None
    seendate = str(article.get("seendate") or "")
    dt = pd.to_datetime(seendate, utc=True, errors="coerce")
    pub_iso = dt.strftime("%Y-%m-%dT%H:%M:%SZ") if dt is not pd.NaT and pd.notna(dt) else ""
    return {
        "url": link,
        "title": title,
        "text_raw": text_raw,
        "published_at_utc": pub_iso,
    }


def _fetch_gdelt_json(url: str, timeout: int, retries: int = 3) -> tuple[dict | None, str | None]:
    last_reason = "unknown"
    for i in range(retries):
        try:
            raw = _fetch(url, timeout=timeout)
            payload = json.loads(raw)
            return payload, None
        except Exception:
            try:
                raw = _fetch(url, timeout=timeout)
                last_reason = (raw or "").strip()[:180] or "empty_response"
            except Exception as e2:
                last_reason = str(e2)[:180]
            if "429" in last_reason.lower() or "too many requests" in last_reason.lower():
                time.sleep(75.0)
            else:
                time.sleep(1.5 * (i + 1))
    return None, last_reason


def fetch_gdelt(
    out_csv: Path,
    checkpoint_csv: Path,
    skip_log_json: Path,
    start_date: str,
    end_date: str,
    max_records: int,
    timeout: int,
    domains: list[str],
    target_total: int,
    per_batch_limit: int,
    max_workers: int,
    request_delay_sec: float,
    checkpoint_every_batches: int,
    progress_json: Path,
) -> dict:
    existing = _read_existing(out_csv)
    known_urls = {r.get("url", "").strip() for r in existing if r.get("url")}
    known_hash = {
        hashlib.md5((r.get("text_raw", "") or "").strip().encode("utf-8")).hexdigest()
        for r in existing
        if (r.get("text_raw", "") or "").strip()
    }
    rows = list(existing)
    checkpoints: list[dict] = []
    skipped: list[dict] = []
    now_iso = datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
    scanned = 0
    added = 0
    batches = 0
    started = time.time()
    last_progress = started

    domains = [d for d in domains if d in SOURCE_MAP]
    if not domains:
        domains = ["rbc.ru", "tass.ru"]
    for m_start, m_end in _month_ranges(start_date, end_date):
        if len(rows) >= target_total:
            break
        for kw in KEYWORDS:
            if len(rows) >= target_total:
                break
            rotating_domains = list(domains)
            random.shuffle(rotating_domains)
            for dom in rotating_domains:
                if len(rows) >= target_total:
                    break
                q = _gdelt_query(kw, dom)
                url = (
                    f"{GDELT_DOC_URL}?query={quote_plus(q)}&mode=ArtList"
                    f"&maxrecords={max_records}&format=json"
                    f"&startdatetime={_to_gdelt_dt(m_start)}&enddatetime={_to_gdelt_dt(m_end)}"
                )
                payload, fetch_reason = _fetch_gdelt_json(url, timeout=timeout, retries=3)
                if payload is None:
                    skipped.append(
                        {
                            "month": m_start.strftime("%Y-%m"),
                            "keyword": kw,
                            "domain": dom,
                            "reason": "gdelt_fetch_error",
                            "details": fetch_reason,
                        }
                    )
                    checkpoints.append(
                        {
                            "month": m_start.strftime("%Y-%m"),
                            "keyword": kw,
                            "domain": dom,
                            "articles_returned": 0,
                            "added_rows": 0,
                            "status": "gdelt_fetch_error",
                            "details": fetch_reason,
                        }
                    )
                    batches += 1
                    if batches % max(1, checkpoint_every_batches) == 0:
                        progress = {
                            "progress_batches": batches,
                            "current_total_rows": len(rows),
                            "added_rows_so_far": added,
                            "scanned_links_so_far": scanned,
                            "elapsed_seconds": int(time.time() - started),
                            "target_total": target_total,
                            "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        }
                        pd.DataFrame(checkpoints).to_csv(checkpoint_csv, index=False)
                        skip_log_json.write_text(json.dumps(skipped, ensure_ascii=False, indent=2), encoding="utf-8")
                        progress_json.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
                        print(json.dumps(progress, ensure_ascii=False), flush=True)
                    if request_delay_sec > 0:
                        time.sleep(request_delay_sec)
                    continue
                arts = payload.get("articles") or []
                batch_added = 0
                if not arts:
                    skipped.append(
                        {"month": m_start.strftime("%Y-%m"), "keyword": kw, "domain": dom, "reason": "empty_batch"}
                    )
                    checkpoints.append(
                        {
                            "month": m_start.strftime("%Y-%m"),
                            "keyword": kw,
                            "domain": dom,
                            "articles_returned": 0,
                            "added_rows": 0,
                            "status": "empty_batch",
                        }
                    )
                    batches += 1
                    if batches % max(1, checkpoint_every_batches) == 0:
                        progress = {
                            "progress_batches": batches,
                            "current_total_rows": len(rows),
                            "added_rows_so_far": added,
                            "scanned_links_so_far": scanned,
                            "elapsed_seconds": int(time.time() - started),
                            "target_total": target_total,
                            "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        }
                        pd.DataFrame(checkpoints).to_csv(checkpoint_csv, index=False)
                        skip_log_json.write_text(json.dumps(skipped, ensure_ascii=False, indent=2), encoding="utf-8")
                        progress_json.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
                        print(json.dumps(progress, ensure_ascii=False), flush=True)
                    if request_delay_sec > 0:
                        time.sleep(request_delay_sec)
                    continue
                batch_arts = []
                for a in arts:
                    u = (a.get("url") or "").strip().split("#")[0]
                    if u and u not in known_urls:
                        batch_arts.append(a)
                    if len(batch_arts) >= max(1, per_batch_limit):
                        break
                scanned += len(batch_arts)
                if batch_arts:
                    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
                        candidates = list(ex.map(lambda x: _extract_candidate(x, timeout, 500), batch_arts))
                    for cand in candidates:
                        if not cand:
                            continue
                        link = cand["url"]
                        h = hashlib.md5(cand["text_raw"].encode("utf-8")).hexdigest()
                        if link in known_urls or h in known_hash:
                            continue
                        pub_iso = cand["published_at_utc"] or m_start.strftime("%Y-%m-%dT%H:%M:%SZ")
                        rows.append(
                            {
                                "news_id": f"gdelt_{hashlib.sha1(link.encode('utf-8')).hexdigest()[:12]}",
                                "source": _source_from_url(link),
                                "url": link,
                                "published_at_utc": pub_iso,
                                "title": cand["title"],
                                "text_raw": cand["text_raw"],
                                "rubric": "monetary_policy",
                                "ingestion_ts": now_iso,
                            }
                        )
                        known_urls.add(link)
                        known_hash.add(h)
                        batch_added += 1
                        added += 1
                checkpoints.append(
                    {
                        "month": m_start.strftime("%Y-%m"),
                        "keyword": kw,
                        "domain": dom,
                        "articles_returned": int(len(arts)),
                        "added_rows": int(batch_added),
                    }
                )
                batches += 1
                elapsed = int(time.time() - started)
                progress = {
                    "progress_batches": batches,
                    "current_total_rows": len(rows),
                    "added_rows_so_far": added,
                    "scanned_links_so_far": scanned,
                    "elapsed_seconds": elapsed,
                    "target_total": target_total,
                    "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
                if batches % max(1, checkpoint_every_batches) == 0:
                    rows = sorted(rows, key=lambda r: r.get("published_at_utc", ""))
                    _write_rows(out_csv, rows)
                    pd.DataFrame(checkpoints).to_csv(checkpoint_csv, index=False)
                    skip_log_json.write_text(json.dumps(skipped, ensure_ascii=False, indent=2), encoding="utf-8")
                    progress_json.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(json.dumps(progress, ensure_ascii=False), flush=True)
                if (time.time() - last_progress) >= 3600:
                    progress_json.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(json.dumps(progress, ensure_ascii=False), flush=True)
                    last_progress = time.time()
                if request_delay_sec > 0:
                    time.sleep(request_delay_sec)
    rows = sorted(rows, key=lambda r: r.get("published_at_utc", ""))
    _write_rows(out_csv, rows)
    pd.DataFrame(checkpoints).to_csv(checkpoint_csv, index=False)
    skip_log_json.write_text(json.dumps(skipped, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "news_csv": str(out_csv),
        "scanned_links": int(scanned),
        "added_rows": int(added),
        "total_rows": int(len(rows)),
        "checkpoint_csv": str(checkpoint_csv),
        "skip_log_json": str(skip_log_json),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--news-csv", default="data/source/news_gdelt.csv")
    p.add_argument("--checkpoint-csv", default="reports/gdelt_checkpoint.csv")
    p.add_argument("--skip-log-json", default="reports/gdelt_skipped_batches.json")
    p.add_argument("--start-date", default="2015-01-01")
    p.add_argument("--end-date", default="2026-04-30")
    p.add_argument("--max-records", type=int, default=75)
    p.add_argument("--timeout", type=int, default=20)
    p.add_argument("--domains", nargs="+", default=["rbc.ru", "tass.ru"])
    p.add_argument("--target-total", type=int, default=4000)
    p.add_argument("--per-batch-limit", type=int, default=8)
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--request-delay-sec", type=float, default=1.2)
    p.add_argument("--checkpoint-every-batches", type=int, default=20)
    p.add_argument("--progress-json", default="reports/gdelt_progress.json")
    args = p.parse_args()
    report = fetch_gdelt(
        out_csv=Path(args.news_csv),
        checkpoint_csv=Path(args.checkpoint_csv),
        skip_log_json=Path(args.skip_log_json),
        start_date=args.start_date,
        end_date=args.end_date,
        max_records=args.max_records,
        timeout=args.timeout,
        domains=args.domains,
        target_total=args.target_total,
        per_batch_limit=args.per_batch_limit,
        max_workers=args.max_workers,
        request_delay_sec=args.request_delay_sec,
        checkpoint_every_batches=args.checkpoint_every_batches,
        progress_json=Path(args.progress_json),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
