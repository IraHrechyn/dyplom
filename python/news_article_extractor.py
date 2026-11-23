from __future__ import annotations

import ipaddress
import json
import re
import socket
from datetime import datetime
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

STRIP_TAGS = frozenset(
    {"script", "style", "nav", "footer", "header", "aside", "form", "noscript", "button", "svg"}
)

CONTENT_CLASS_HINTS = (
    "news-text",
    "article-text",
    "article__text",
    "post-content",
    "entry-content",
    "article-body",
    "article",
    "post",
    "content",
    "body",
)
PREFERRED_CONTENT_SELECTORS = (
    ".news-text",
    "[itemprop='articleBody']",
    ".article__text",
    ".article-text",
    ".post-content",
    ".entry-content",
    "article",
)

DEFAULT_TIMEOUT = 15
MAX_REDIRECTS = 3
BLOCKED_HOSTS = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "metadata.google.internal",
        "metadata",
    }
)

class NewsExtractionError(Exception):
    pass

def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )

def _assert_public_url(url: str) -> None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").strip().lower().rstrip(".")
    if not host:
        raise NewsExtractionError("Invalid URL: missing host")
    if host in BLOCKED_HOSTS or host.endswith(".localhost") or host.endswith(".local"):
        raise NewsExtractionError("URL host is not allowed")
    if host == "0.0.0.0":
        raise NewsExtractionError("URL host is not allowed")
    try:
        ipaddress.ip_address(host)
        if _is_blocked_ip(ipaddress.ip_address(host)):
            raise NewsExtractionError("URL resolves to a private/reserved address")
        return
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise NewsExtractionError(f"Failed to resolve host: {host}") from e
    if not infos:
        raise NewsExtractionError(f"Failed to resolve host: {host}")
    for info in infos:
        ip_s = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_s)
        except ValueError:
            continue
        if _is_blocked_ip(ip):
            raise NewsExtractionError("URL resolves to a private/reserved address")

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
    _assert_public_url(url)
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

def _container_score(tag: Tag) -> tuple[int, int]:
    ps = tag.find_all("p")
    text_len = len(tag.get_text(" ", strip=True))
    return (len(ps), text_len)


def _find_main_container(soup: BeautifulSoup) -> Tag:
    for sel in PREFERRED_CONTENT_SELECTORS:
        node = soup.select_one(sel)
        if isinstance(node, Tag):
            score = _container_score(node)
            if score[0] >= 2 or score[1] >= 400:
                return node

    article = soup.find("article")
    if isinstance(article, Tag) and _container_score(article)[1] >= 400:
        return article

    best: Tag | None = None
    best_score = (-1, -1)
    for hint in CONTENT_CLASS_HINTS:
        for div in soup.find_all("div", class_=True):
            if not _class_has_hint(div.get("class"), (hint,)):
                continue
            score = _container_score(div)
            if score > best_score:
                best_score = score
                best = div
    if best is not None and best_score[1] >= 400:
        return best

    for tag in soup.find_all(["div", "section"]):
        if not isinstance(tag, Tag):
            continue
        score = _container_score(tag)
        if score[0] >= 4 and score > best_score:
            best_score = score
            best = tag
    if best is not None and best_score[1] >= 400:
        return best

    main = soup.find("main")
    if isinstance(main, Tag) and _container_score(main)[1] >= 400:
        return main

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
    return re.sub(r"\s+", " ", (raw or "")).strip()

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

def _walk_json_ld_for_date(obj: Any, *, allow_modified: bool = False) -> str | None:
    publish_keys = ("datePublished", "dateCreated", "uploadDate", "publicationDate")
    modified_keys = ("dateModified", "dateUpdated")
    keys = publish_keys + (modified_keys if allow_modified else ())
    if isinstance(obj, dict):
        for key in keys:
            if key in obj:
                got = _normalize_date_iso(str(obj[key]))
                if got:
                    return got
        for v in obj.values():
            got = _walk_json_ld_for_date(v, allow_modified=allow_modified)
            if got:
                return got
    elif isinstance(obj, list):
        for item in obj:
            got = _walk_json_ld_for_date(item, allow_modified=allow_modified)
            if got:
                return got
    return None

def _json_ld_dates(soup: BeautifulSoup, *, allow_modified: bool) -> str | None:
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
            got = _walk_json_ld_for_date(item, allow_modified=allow_modified)
            if got:
                return got
    return None

def extract_published_date_from_soup(soup: BeautifulSoup) -> str | None:
    for prop in ("article:published_time", "og:published_time"):
        tag = soup.find("meta", attrs={"property": prop}) or soup.find("meta", attrs={"property": prop.lower()})
        if tag and tag.get("content"):
            got = _normalize_date_iso(tag["content"])
            if got:
                return got

    for name in ("pubdate", "publishdate", "originalpublicationdate", "date"):
        tag = soup.find("meta", attrs={"name": re.compile("^" + name + "$", re.I)})
        if tag and tag.get("content"):
            got = _normalize_date_iso(tag["content"])
            if got:
                return got

    for t in soup.find_all("time"):
        itemprop = (t.get("itemprop") or "").lower()
        if itemprop and itemprop not in ("datepublished", "datecreated", "pubdate"):
            continue
        dt_attr = t.get("datetime")
        if dt_attr:
            got = _normalize_date_iso(dt_attr)
            if got:
                return got

    got = _json_ld_dates(soup, allow_modified=False)
    if got:
        return got

    for prop in ("article:modified_time", "og:updated_time"):
        tag = soup.find("meta", attrs={"property": prop}) or soup.find("meta", attrs={"property": prop.lower()})
        if tag and tag.get("content"):
            got = _normalize_date_iso(tag["content"])
            if got:
                return got

    return _json_ld_dates(soup, allow_modified=True)

def _browser_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "uk-UA,uk;q=0.9,en-US;q=0.8,en;q=0.7",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Upgrade-Insecure-Requests": "1",
    }


def _looks_like_bot_challenge(html: str) -> bool:
    head = (html or "")[:5000].lower()
    markers = (
        "необхідно виконати дію",
        "just a moment",
        "cf-challenge",
        "attention required",
        "checking your browser",
        "enable javascript and cookies",
    )
    return any(m in head for m in markers)


class _HtmlResponse:
    def __init__(self, *, url: str, text: str, encoding: str = "utf-8"):
        self.url = url
        self.text = text
        self.encoding = encoding
        self.apparent_encoding = encoding


def _fetch_with_curl_cffi(url: str, timeout: int) -> _HtmlResponse:
    from curl_cffi import requests as creq

    impersonations = (
        "safari17_0",
        "chrome131",
        "chrome124",
        "chrome120",
        "chrome110",
    )
    last_err: Exception | None = None
    for impersonate in impersonations:
        current = url
        try:
            for _ in range(MAX_REDIRECTS + 1):
                _assert_public_url(current)
                response = creq.get(
                    current,
                    timeout=timeout,
                    headers=_browser_headers(),
                    allow_redirects=False,
                    impersonate=impersonate,
                )
                if response.status_code in (301, 302, 303, 307, 308):
                    loc = response.headers.get("Location") or response.headers.get("location")
                    if not loc:
                        raise NewsExtractionError("Redirect without Location header")
                    current = urljoin(current, loc)
                    continue
                if response.status_code >= 400:
                    raise NewsExtractionError(f"HTTP {response.status_code} while fetching article")
                text = response.text or ""
                if _looks_like_bot_challenge(text):
                    raise NewsExtractionError("bot_challenge")
                final_url = str(getattr(response, "url", None) or current)
                _assert_public_url(final_url)
                return _HtmlResponse(url=final_url, text=text, encoding="utf-8")
            raise NewsExtractionError("Too many redirects")
        except NewsExtractionError as e:
            last_err = e
            if str(e) == "bot_challenge" or "HTTP 403" in str(e) or "HTTP 429" in str(e):
                continue
            raise
        except Exception as e:
            last_err = e
            continue
    raise NewsExtractionError(
        "Сайт захищено від автоматичного зчитування. "
        "Скопіюйте текст статті вручну або надішліть скріншот."
    ) from last_err


def _fetch_with_requests(url: str, timeout: int) -> _HtmlResponse:
    session = requests.Session()
    session.max_redirects = MAX_REDIRECTS
    headers = _browser_headers()
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        _assert_public_url(current)
        try:
            response = session.get(
                current,
                timeout=timeout,
                headers=headers,
                allow_redirects=False,
            )
        except requests.Timeout as e:
            raise NewsExtractionError(f"Request timed out after {timeout}s") from e
        except requests.RequestException as e:
            raise NewsExtractionError(f"Failed to fetch URL: {e}") from e
        if response.is_redirect or response.status_code in (301, 302, 303, 307, 308):
            loc = response.headers.get("Location")
            if not loc:
                raise NewsExtractionError("Redirect without Location header")
            current = urljoin(current, loc)
            continue
        if response.status_code >= 400:
            raise NewsExtractionError(
                f"HTTP {response.status_code}: сайт не віддав статтю. "
                "Спробуйте вставити текст вручну або надіслати скріншот."
            )
        if not response.encoding or response.encoding.lower() in ("iso-8859-1", "ascii"):
            response.encoding = response.apparent_encoding or "utf-8"
        text = response.text or ""
        if _looks_like_bot_challenge(text):
            raise NewsExtractionError(
                "Сайт захищено від автоматичного зчитування. "
                "Скопіюйте текст статті вручну або надішліть скріншот."
            )
        final_url = str(response.url)
        _assert_public_url(final_url)
        return _HtmlResponse(url=final_url, text=text, encoding=response.encoding or "utf-8")
    raise NewsExtractionError("Too many redirects")


def _fetch_public_html(url: str, timeout: int) -> _HtmlResponse:
    try:
        return _fetch_with_curl_cffi(url, timeout)
    except ImportError:
        return _fetch_with_requests(url, timeout)
    except NewsExtractionError:
        raise
    except Exception:
        return _fetch_with_requests(url, timeout)

def extract_news_article(url: str, timeout: int = DEFAULT_TIMEOUT) -> dict[str, object]:
    normalized = _normalize_url(url)
    response = _fetch_public_html(normalized, timeout=timeout or DEFAULT_TIMEOUT)
    soup = BeautifulSoup(response.text, "html.parser")
    base = str(response.url)

    published_at = extract_published_date_from_soup(soup)

    container = _find_main_container(soup)
    if not isinstance(container, Tag):
        raise NewsExtractionError("Could not locate any HTML container")

    _strip_noise_tags(container)

    raw_text = container.get_text(separator=" ", strip=True)
    text = _clean_article_text(raw_text)
    if _looks_like_bot_challenge(text) or len(text) < 120:
        raise NewsExtractionError(
            "Не вдалося зчитати текст статті з цієї сторінки "
            "(сайт може блокувати автоматичний доступ). "
            "Скопіюйте текст новини в поле вручну або надішліть скріншот."
        )

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
