from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

STRIP_TAGS = frozenset({"script", "style", "nav", "footer", "header", "aside", "form", "noscript"})

CONTENT_CLASS_HINTS = ("content", "article", "post", "body")

UI_SHORT_STOPWORDS = frozenset(
    {
        "home",
        "login",
        "logout",
        "search",
        "menu",
        "share",
        "more",
        "next",
        "prev",
        "back",
        "skip",
    }
)

DEFAULT_TIMEOUT = 15

class NewsExtractionError(Exception):
    pass

def _normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        raise NewsExtractionError("URL is empty")
    parsed = urlparse(url)
    if not parsed.scheme:
        url = "https://" + url
        parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise NewsExtractionError(f"Unsupported URL scheme: {parsed.scheme!r}")
    if not parsed.netloc:
        raise NewsExtractionError("Invalid URL: missing host")
    return url

def _class_has_hint(class_attr: str | list[str] | None, hints: Iterable[str]) -> bool:
    if not class_attr:
        return False
    if isinstance(class_attr, str):
        parts = re.split(r"\s+", class_attr.strip())
    else:
        parts = class_attr
    lowered = [p.lower() for p in parts if p]
    for hint in hints:
        for p in lowered:
            if hint in p:
                return True
    return False

def _find_main_container(soup: BeautifulSoup) -> Tag:
    article = soup.find("article")
    if article:
        return article

    main = soup.find("main")
    if main:
        return main

    for hint in CONTENT_CLASS_HINTS:
        for div in soup.find_all("div", class_=True):
            if _class_has_hint(div.get("class"), (hint,)):
                return div

    body = soup.find("body")
    if body:
        return body
    return soup

def _strip_noise_tags(root: Tag) -> None:
    for tag in list(root.find_all(STRIP_TAGS)):
        tag.decompose()

def _to_absolute(base: str, href: str | None) -> str | None:
    if not href or not href.strip():
        return None
    href = href.strip()
    if href.startswith(("data:", "javascript:", "mailto:", "tel:", "#")):
        return None
    return urljoin(base, href)

def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _clean_article_text(raw: str) -> str:
    raw = re.sub(r"\s+", " ", (raw or "")).strip()
    if not raw:
        return ""
    tokens = raw.split()
    filtered: list[str] = []
    for t in tokens:
        tl = re.sub(r"^[^\w]+|[^\w]+$", "", t).lower()
        if tl in UI_SHORT_STOPWORDS:
            continue
        filtered.append(t)
    return " ".join(filtered)

def _collect_images(container: Tag, base_url: str) -> list[str]:
    urls: list[str] = []
    for img in container.find_all("img"):
        src = img.get("src") or img.get("data-src")
        abs_url = _to_absolute(base_url, src)
        if abs_url:
            urls.append(abs_url)
    return _dedupe_preserve_order(urls)

def _collect_videos(container: Tag, base_url: str) -> list[str]:
    urls: list[str] = []

    for video in container.find_all("video"):
        vsrc = video.get("src")
        if vsrc:
            u = _to_absolute(base_url, vsrc)
            if u:
                urls.append(u)
        for src in video.find_all("source"):
            s = src.get("src")
            u = _to_absolute(base_url, s)
            if u:
                urls.append(u)

    for iframe in container.find_all("iframe"):
        src = iframe.get("src")
        u = _to_absolute(base_url, src)
        if u:
            urls.append(u)

    return _dedupe_preserve_order(urls)

def _normalize_date_iso(raw: str | None) -> str | None:
    if not raw or not str(raw).strip():
        return None
    s = str(raw).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        try:
            datetime.strptime(s[:10], "%Y-%m-%d")
            return s[:10]
        except ValueError:
            pass
    try:
        dt = datetime.fromisoformat(s.replace(" ", "T"))
        return dt.date().isoformat()
    except ValueError:
        pass
    m = re.search(r"(\d{4}-\d{2}-\d{2})", s)
    if m:
        try:
            datetime.strptime(m.group(1), "%Y-%m-%d")
            return m.group(1)
        except ValueError:
            pass
    return None

def _walk_json_ld_for_date(obj: Any) -> str | None:
    if isinstance(obj, dict):
        for key in ("datePublished", "dateCreated", "uploadDate", "publicationDate"):
            if key in obj:
                got = _normalize_date_iso(str(obj[key]))
                if got:
                    return got
        for v in obj.values():
            got = _walk_json_ld_for_date(v)
            if got:
                return got
    elif isinstance(obj, list):
        for item in obj:
            got = _walk_json_ld_for_date(item)
            if got:
                return got
    return None

def extract_published_date_from_soup(soup: BeautifulSoup) -> str | None:
    for prop in (
        "article:published_time",
        "article:modified_time",
        "og:published_time",
        "og:updated_time",
    ):
        tag = soup.find("meta", attrs={"property": prop}) or soup.find("meta", attrs={"property": prop.lower()})
        if tag and tag.get("content"):
            got = _normalize_date_iso(tag["content"])
            if got:
                return got

    for name in ("pubdate", "publishdate", "date", "originalpublicationdate"):
        tag = soup.find("meta", attrs={"name": re.compile("^" + name + "$", re.I)})
        if tag and tag.get("content"):
            got = _normalize_date_iso(tag["content"])
            if got:
                return got

    for t in soup.find_all("time"):
        dt_attr = t.get("datetime")
        if dt_attr:
            got = _normalize_date_iso(dt_attr)
            if got:
                return got

    for script in soup.find_all("script", attrs={"type": re.compile("application/ld\\+json", re.I)}):
        raw = (script.string or script.get_text() or "").strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "@graph" in data:
            data = data["@graph"]
        items = data if isinstance(data, list) else [data]
        for item in items:
            got = _walk_json_ld_for_date(item)
            if got:
                return got

    return None

def extract_news_article(url: str, timeout: int = DEFAULT_TIMEOUT) -> dict[str, object]:
    normalized = _normalize_url(url)

    try:
        response = requests.get(
            normalized,
            timeout=timeout,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        response.raise_for_status()
    except requests.Timeout as e:
        raise NewsExtractionError(f"Request timed out after {timeout}s") from e
    except requests.RequestException as e:
        raise NewsExtractionError(f"Failed to fetch URL: {e}") from e

    if not response.encoding:
        response.encoding = response.apparent_encoding

    soup = BeautifulSoup(response.text, "html.parser")
    base = str(response.url)

    published_at = extract_published_date_from_soup(soup)

    container = _find_main_container(soup)
    if not isinstance(container, Tag):
        raise NewsExtractionError("Could not locate any HTML container")

    _strip_noise_tags(container)

    raw_text = container.get_text(separator=" ", strip=True)
    text = _clean_article_text(raw_text)

    images = _collect_images(container, base)
    videos = _collect_videos(container, base)

    return {
        "text": text,
        "images": images,
        "videos": videos,
        "publishedAt": published_at,
    }

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Зчитати статтю з URL і вивести в консоль.")
    parser.add_argument("url", nargs="?", help="Адреса сторінки новин")
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Таймаут запиту, с (за замовчуванням {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Лише JSON у stdout (для API Next.js)",
    )
    args = parser.parse_args()
    if not args.url:
        print("Вкажіть URL: python news_article_extractor.py <url>", file=sys.stderr)
        sys.exit(1)

    try:
        data = extract_news_article(args.url, timeout=args.timeout)
    except NewsExtractionError as e:
        if args.json:
            print(json.dumps({"error": str(e)}, ensure_ascii=False))
        else:
            print(f"Помилка: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(data, ensure_ascii=False))
        sys.exit(0)

    print("— ТЕКСТ —")
    print(data.get("text") or "(порожньо)")
    print()
    print("— ЗОБРАЖЕННЯ —")
    for i, u in enumerate(data.get("images") or [], 1):
        print(f"  {i}. {u}")
    print()
    print("— ВІДЕО / IFRAME —")
    for i, u in enumerate(data.get("videos") or [], 1):
        print(f"  {i}. {u}")
    print()
    print("— JSON (повний результат) —")
    print(json.dumps(data, ensure_ascii=False, indent=2))
