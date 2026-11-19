from __future__ import annotations

import os
import re
from typing import Any
from urllib.parse import quote, urlparse

import requests

DEFAULT_TIMEOUT = 12
MAX_ARTICLES = 12

def _search_query_from_text(text: str, max_words: int = 14, max_len: int = 120) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    t = re.sub(r"https?://\S+", " ", t)
    words = [w.strip(".,!?;:()[]\"'«»—") for w in t.split()]
    words = [w for w in words if len(w) > 2 and not w.isdigit()]
    q = " ".join(words[:max_words]).strip()
    if len(q) > max_len:
        q = q[:max_len].rsplit(" ", 1)[0]
    return q

def _keyword_query(text: str, max_terms: int = 8) -> str:
    uk_stop = {
        "і", "та", "або", "але", "що", "це", "як", "для", "про", "від", "до", "на", "по",
        "зі", "із", "не", "є", "був", "була", "було", "були", "цей", "ця", "ті", "той",
        "вже", "ще", "дуже", "також", "якщо", "щоб", "теж", "буде", "після", "перед",
        "може", "можливе", "повідомляє", "повідомив", "заявив", "сказав",
    }
    en_stop = {
        "the", "a", "an", "and", "or", "but", "that", "this", "for", "from", "with",
        "into", "about", "not", "is", "are", "was", "were", "be", "it", "its", "of",
        "to", "in", "on", "at", "by", "as", "said", "says", "after", "before",
    }
    words = re.findall(r"[\w\-]+", text or "", flags=re.UNICODE)
    scored: list[tuple[int, str]] = []
    seen: set[str] = set()
    for raw in words:
        lower = raw.lower()
        if len(lower) < 3 or lower in uk_stop or lower in en_stop or lower in seen:
            continue
        seen.add(lower)
        score = min(len(raw), 12)
        if raw[:1].isupper():
            score += 8
        if raw.isupper() and len(raw) >= 2:
            score += 12
        if re.search(r"[A-Za-z]", raw):
            score += 6
        if re.fullmatch(r"\d{4}", raw):
            score += 5
        scored.append((score, raw))
    scored.sort(key=lambda x: -x[0])
    return " ".join(w for _, w in scored[:max_terms]).strip()

def _latin_query(text: str, max_terms: int = 6) -> str:
    words = re.findall(r"[A-Za-z][\w.\-]*", text or "")
    en_stop = {
        "the", "a", "an", "and", "or", "but", "that", "this", "for", "from", "with",
        "of", "to", "in", "on",
    }
    out: list[str] = []
    seen: set[str] = set()
    for w in words:
        key = w.lower()
        if len(w) < 2 or key in en_stop or key in seen:
            continue
        seen.add(key)
        out.append(w)
        if len(out) >= max_terms:
            break
    return " ".join(out)

def _entity_queries(text: str) -> list[str]:
    words = re.findall(r"[\w\-]+", text or "", flags=re.UNICODE)
    entities: list[str] = []
    seen: set[str] = set()
    for i, raw in enumerate(words):
        if len(raw) < 3:
            continue
        is_name = raw[:1].isupper() and not raw.isupper()
        is_latin = bool(re.search(r"[A-Za-z]", raw))
        if not (is_name or is_latin):
            continue
        key = raw.lower()
        if key in seen:
            continue
        seen.add(key)
        entities.append(raw)
        if i + 1 < len(words):
            nxt = words[i + 1]
            if nxt[:1].isupper() and len(nxt) >= 3:
                pair = f"{raw} {nxt}"
                pk = pair.lower()
                if pk not in seen:
                    seen.add(pk)
                    entities.append(pair)
        if len(entities) >= 10:
            break

    out: list[str] = []
    for e in entities:
        if len(e.split()) >= 2:
            out.append(e)
    for e in entities:
        if e not in out and re.search(r"[A-Za-zА-Яа-яЇїІіЄєҐґ]", e):
            out.append(e)
        if len(out) >= 6:
            break
    return out

def _uniq_queries(*parts: str, limit: int = 8) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        t = re.sub(r"\s+", " ", (p or "").strip())
        if len(t) < 3:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= limit:
            break
    return out

def _wiki_search(lang: str, query: str, limit: int, timeout: float) -> list[dict[str, Any]]:
    if not query:
        return []
    api = f"https://{lang}.wikipedia.org/w/api.php"
    r = requests.get(
        api,
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
            "formatversion": 2,
        },
        timeout=timeout,
        headers={
            "User-Agent": "DyplomNewsAnalyzer/1.0 (university project; Python requests)",
            "Accept": "application/json",
        },
    )
    r.raise_for_status()
    data = r.json()
    hits = (data.get("query") or {}).get("search") or []
    base = f"https://{lang}.wikipedia.org/wiki/"
    out: list[dict[str, Any]] = []
    for h in hits:
        title = h.get("title") or ""
        if not title:
            continue
        slug = quote(title.replace(" ", "_"), safe="()%")
        url = base + slug
        snippet = re.sub(r"<[^>]+>", "", (h.get("snippet") or ""))
        out.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet[:280] + ("…" if len(snippet) > 280 else ""),
                "publisher": f"Вікіпедія ({lang.upper()})",
            }
        )
    return out

UK_NEWS_DOMAINS = (
    "pravda.com.ua,unn.com.ua,ukrinform.ua,suspilne.media,censor.net,lb.ua,"
    "nv.ua,24tv.ua,tsn.ua,unian.net,interfax.com.ua,radiosvoboda.org,"
    "football24.ua,ua-football.com,sport.ua,tribuna.com"
)

def _newsapi_fetch(query: str, limit: int, timeout: float, **extra: str) -> list[dict[str, Any]]:
    key = (os.environ.get("NEWS_API_KEY") or "").strip()
    if not key or not query:
        return []
    try:
        params: dict[str, Any] = {
            "q": query,
            "sortBy": "relevancy",
            "pageSize": min(limit, 10),
            **extra,
        }
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params=params,
            headers={"X-Api-Key": key},
            timeout=timeout,
        )
        if r.status_code != 200:
            return []
        data = r.json()
        out: list[dict[str, Any]] = []
        for art in (data.get("articles") or [])[:limit]:
            title = (art.get("title") or "").strip()
            url = (art.get("url") or "").strip()
            if not title or not url or not urlparse(url).scheme.startswith("http"):
                continue
            desc = (art.get("description") or art.get("content") or "")[:300]
            src = art.get("source")
            pub = (
                src.get("name")
                if isinstance(src, dict)
                else (str(src) if src else "NewsAPI")
            ) or "NewsAPI"
            out.append({"title": title, "url": url, "snippet": desc, "publisher": pub})
        return out
    except requests.RequestException:
        return []

def _newsapi_search(query: str, limit: int, timeout: float) -> list[dict[str, Any]]:
    per = max(3, (limit + 2) // 3)
    batches = [
        _newsapi_fetch(query, per, timeout, language="uk"),
        _newsapi_fetch(query, per, timeout, language="en"),
        _newsapi_fetch(query, per, timeout),
    ]
    return batches[0] + batches[1] + batches[2]

def _google_news_rss(query: str, hl: str, gl: str, ceid: str, limit: int, timeout: float) -> list[dict[str, Any]]:
    if not query:
        return []
    try:
        url = (
            "https://news.google.com/rss/search?"
            f"q={quote(query)}&hl={hl}&gl={gl}&ceid={ceid}"
        )
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "DyplomNewsAnalyzer/1.0"})
        if r.status_code != 200:
            return []
        xml = r.text
        out: list[dict[str, Any]] = []
        for block in re.findall(r"<item>([\s\S]*?)</item>", xml):
            if len(out) >= limit:
                break
            title_m = re.search(r"<title>(?:<!\[CDATA\[)?([\s\S]*?)(?:\]\]>)?</title>", block, re.I)
            link_m = re.search(r"<link>([\s\S]*?)</link>", block, re.I)
            src_m = re.search(r"<source[^>]*>([\s\S]*?)</source>", block, re.I)
            title = (title_m.group(1) if title_m else "").strip()
            link = (link_m.group(1) if link_m else "").strip()
            pub = (src_m.group(1) if src_m else "Google News").strip()
            if link.startswith("http") and title:
                out.append({"title": title, "url": link, "snippet": title, "publisher": pub})
        return out
    except requests.RequestException:
        return []

def _overlap_score(title: str, needles: list[str]) -> int:
    t = (title or "").lower()
    score = 0
    for n in needles:
        token = n.lower().strip()
        if len(token) < 4:
            continue
        if token in t:
            score += 3 if " " in token else 1
    return score

def _filter_relevant(items: list[dict[str, Any]], needles: list[str]) -> list[dict[str, Any]]:
    if not items or not needles:
        return items
    ranked = sorted(
        items,
        key=lambda a: _overlap_score(str(a.get("title") or ""), needles),
        reverse=True,
    )
    strong = [a for a in ranked if _overlap_score(str(a.get("title") or ""), needles) > 0]
    return strong if strong else ranked

def _add_items(
    combined: list[dict[str, Any]],
    seen_urls: set[str],
    items: list[dict[str, Any]],
    limit: int,
    needles: list[str] | None = None,
) -> bool:
    ordered = _filter_relevant(items, needles or [])
    for item in ordered:
        u = (item.get("url") or "").split("?", 1)[0]
        if not u or u in seen_urls:
            continue
        seen_urls.add(u)
        combined.append(item)
        if len(combined) >= limit:
            return True
    return False

def fetch_related_reading_payload(
    text: str,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    q = _search_query_from_text(text)
    short_q = _keyword_query(text, 8) or _search_query_from_text(text, max_words=8)
    latin_q = _latin_query(text, 6)
    entities = _entity_queries(text)

    queries = _uniq_queries(
        short_q,
        " ".join(short_q.split()[:4]),
        *entities[:4],
        latin_q,
        " ".join(latin_q.split()[:3]) if latin_q else "",
        q,
        limit=8,
    )
    if not queries and len(q) >= 8:
        queries = [q]
    if not queries:
        return {"relatedArticles": [], "readingSearchQuery": short_q or q}

    primary_q = short_q or queries[0]
    needles = _uniq_queries(*entities, *short_q.split()[:6], *latin_q.split()[:4], limit=12)
    seen_urls: set[str] = set()
    combined: list[dict[str, Any]] = []
    t = float(timeout)

    locales = [
        ("uk", "UA", "UA:uk"),
        ("en", "UA", "UA:en"),
        ("en", "GB", "GB:en"),
        ("en", "US", "US:en"),
    ]
    prioritized = _uniq_queries(*entities[:4], *queries[:5], limit=7)
    for query in prioritized:
        for hl, gl, ceid in locales:
            if _add_items(
                combined,
                seen_urls,
                _google_news_rss(query, hl, gl, ceid, 6, t),
                MAX_ARTICLES,
                needles,
            ):
                return {"relatedArticles": combined, "readingSearchQuery": primary_q}

    for query in prioritized[:5]:
        if _add_items(
            combined,
            seen_urls,
            _newsapi_fetch(query, 6, t, domains=UK_NEWS_DOMAINS),
            MAX_ARTICLES,
            needles,
        ):
            return {"relatedArticles": combined, "readingSearchQuery": primary_q}
        if _add_items(
            combined, seen_urls, _newsapi_search(query, 6, t), MAX_ARTICLES, needles
        ):
            return {"relatedArticles": combined, "readingSearchQuery": primary_q}

    for query in prioritized[:4]:
        for lang in ("uk", "en"):
            need = MAX_ARTICLES - len(combined)
            if need <= 0:
                break
            if _add_items(
                combined,
                seen_urls,
                _wiki_search(lang, query, min(need + 2, 8), t),
                MAX_ARTICLES,
                needles,
            ):
                return {"relatedArticles": combined, "readingSearchQuery": primary_q}

    return {"relatedArticles": combined, "readingSearchQuery": primary_q}

def get_related_reading(text: str, timeout: float = DEFAULT_TIMEOUT) -> list[dict[str, Any]]:
    return fetch_related_reading_payload(text, timeout)["relatedArticles"]
