from __future__ import annotations

import os
import re
from typing import Any

import requests

TRUST = {
    "openrouter": 0.20,
    "gemini": 0.20,
    "style": 0.60,
}

def factcheck_api_url() -> str:
    return (os.environ.get("FACTCHECK_API_URL") or "http://127.0.0.1:3000/api/factcheck").strip()

def call_factcheck(
    text: str,
    *,
    verification_mode: str = "publication",
    published_at: str | None = None,
    reference_date: str | None = None,
    timeout: float = 300.0,
    fast: bool = False,
    providers: list[str] | None = None,
) -> dict[str, Any]:
    payload_text = (text or "").strip()
    if len(payload_text) < 15:
        return {"error": "Текст занадто короткий для фактчеку", "factChecks": [], "consensus": None}

    mode = "today" if str(verification_mode).strip().lower() == "today" else "publication"
    payload: dict[str, Any] = {
        "text": payload_text,
        "verificationMode": mode,
        "publishedAt": published_at,
        "referenceDate": reference_date,
        "skipConsensus": True,
    }
    if providers:
        payload["providers"] = providers
    elif fast:
        payload["providers"] = ["openrouter", "gemini"]

    try:
        r = requests.post(
            factcheck_api_url(),
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
    except requests.RequestException as e:
        return {
            "error": (
                f"Не вдалося звернутись до фактчеку ({factcheck_api_url()}). "
                f"Переконайтесь, що веб-сервер запущено (npm run dev). Деталі: {e}"
            ),
            "factChecks": [],
            "consensus": None,
        }

    try:
        data = r.json()
    except Exception:
        return {
            "error": f"Фактчек повернув не-JSON (HTTP {r.status_code})",
            "factChecks": [],
            "consensus": None,
        }

    if not r.ok or data.get("error"):
        return {
            "error": str(data.get("error") or f"HTTP {r.status_code}"),
            "factChecks": [],
            "consensus": None,
        }

    checks = data.get("factChecks")
    if not isinstance(checks, list) or not checks:
        checks = [data] if data.get("overall") or data.get("claims") else []

    consensus = data.get("consensus")
    if not isinstance(consensus, dict):
        consensus = None

    return {
        "factChecks": checks,
        "consensus": consensus,
        "errors": data.get("errors"),
        "error": None,
    }

def soft_vote_combined(style_result: dict[str, Any], fact_checks: list[dict[str, Any]]) -> dict[str, Any]:
    ensemble_fake = float(style_result.get("fakePercentage") or 0)
    mb = style_result.get("modelBreakdown") or {}
    votes: list[tuple[str, float, float]] = []

    style_w = TRUST["style"]
    active = str(mb.get("activeModel") or "")
    sk_w = float(mb.get("sklearnWeight") or 0)
    ke_w = float(mb.get("kerasWeight") or 0)
    if active == "tfidf" or sk_w > ke_w:
        votes.append(("TF-IDF (UA)", ensemble_fake, style_w))
    elif active == "keras" or ke_w > 0 or mb:
        votes.append(("Keras (EN)", ensemble_fake, style_w))
    else:
        votes.append(("стиль", ensemble_fake, style_w))

    usable = []
    for fc in fact_checks:
        o = fc.get("overall") or {}
        if not isinstance(o, dict):
            continue
        fake = o.get("fakePercent")
        if not isinstance(fake, (int, float)):
            continue
        has_sources = bool(fc.get("sourcesUsed"))
        if fc.get("plausibilityOnly") and not has_sources and o.get("verdict") == "NOT_ENOUGH_INFO":
            continue
        usable.append(fc)

    for fc in usable:
        p = str(fc.get("provider") or "").lower()
        if "gemini" in p:
            w = TRUST["gemini"]
        elif "openrouter" in p:
            w = TRUST["openrouter"]
        else:
            w = 0.35
        if (fc.get("overall") or {}).get("verdict") == "NOT_ENOUGH_INFO":
            w *= 0.7
        votes.append((str(fc.get("provider") or "LLM"), float(fc["overall"]["fakePercent"]), w))

    if not votes:
        return {
            "combinedFakePercent": ensemble_fake,
            "combinedTruthPercent": round(100.0 - ensemble_fake, 1),
            "note": "Фінальна оцінка за стилем тексту.",
        }

    sum_w = sum(w for _, _, w in votes) or 1.0
    combined_fake = round(sum(fake * w for _, fake, w in votes) / sum_w, 1)
    if usable:
        note = "Фінальна оцінка за результатами повної перевірки."
    elif fact_checks:
        note = "Фактчек виконано; на діаграмі зараз переважно стильова оцінка."
    else:
        note = "Фінальна оцінка за стилем тексту."
    return {
        "combinedFakePercent": combined_fake,
        "combinedTruthPercent": round(100.0 - combined_fake, 1),
        "note": note,
    }

def _claim_tokens(claim: str) -> set[str]:
    return {w for w in re.findall(r"[\w\-]+", (claim or "").lower(), flags=re.UNICODE) if len(w) >= 4}

def _claims_near_duplicate(a: str, b: str) -> bool:
    ta, tb = _claim_tokens(a), _claim_tokens(b)
    if not ta or not tb:
        return False
    inter = len(ta & tb)
    union = len(ta | tb) or 1
    jaccard = inter / union
    return jaccard >= 0.55 or inter / len(ta) >= 0.75 or inter / len(tb) >= 0.75

def _dedupe_claims(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_items = sorted(items, key=lambda x: -float(x.get("confidence") or 0))
    out: list[dict[str, Any]] = []
    for c in sorted_items:
        if any(_claims_near_duplicate(str(x.get("claim") or ""), str(c.get("claim") or "")) for x in out):
            continue
        out.append(c)
    return out

def merge_claims(
    fact_checks: list[dict[str, Any]],
    *,
    true_limit: int = 6,
    false_limit: int = 4,
) -> dict[str, list[dict[str, Any]]]:
    supported: list[dict[str, Any]] = []
    contradicted: list[dict[str, Any]] = []

    for fc in fact_checks:
        for c in fc.get("claims") or []:
            if not isinstance(c, dict):
                continue
            claim = str(c.get("claim") or "").strip()
            if not claim:
                continue
            verdict = c.get("verdict")
            conf = float(c.get("confidence") or 0)
            if verdict == "SUPPORTED":
                conf = max(conf, 0.9 if c.get("citations") else 0.85)
            elif verdict == "CONTRADICTED":
                conf = max(conf, 0.88 if c.get("citations") else 0.85)
            else:
                continue
            item = {
                "claim": claim,
                "confidence": min(1.0, conf),
                "note": str(c.get("note") or "").strip() or None,
            }
            if verdict == "SUPPORTED":
                supported.append(item)
            else:
                contradicted.append(item)

    supported_ok = [c for c in _dedupe_claims(supported) if float(c["confidence"]) >= 0.85]
    contradicted_ok = [c for c in _dedupe_claims(contradicted) if float(c["confidence"]) >= 0.85]
    return {
        "supported": (supported_ok or _dedupe_claims(supported))[:true_limit],
        "contradicted": contradicted_ok[:false_limit],
    }

def enrich_like_web(style_result: dict[str, Any]) -> dict[str, Any]:
    text = str(style_result.get("text") or "").strip()
    fc = call_factcheck(
        text,
        verification_mode=str(style_result.get("verificationMode") or "publication"),
        published_at=(str(style_result["publishedAt"]).strip() if style_result.get("publishedAt") else None),
        reference_date=(
            str(style_result["referenceDate"]).strip() if style_result.get("referenceDate") else None
        ),
        timeout=90.0,
        fast=True,
    )
    checks = fc.get("factChecks") or []
    combined = soft_vote_combined(style_result, checks)
    claims = merge_claims(checks)

    out = dict(style_result)
    out["truthPercentage"] = combined["combinedTruthPercent"]
    out["fakePercentage"] = combined["combinedFakePercent"]
    out["combined"] = combined
    out["factChecks"] = checks
    out["factCheckConsensus"] = fc.get("consensus")
    out["factCheckError"] = fc.get("error")
    if fc.get("errors") and not out["factCheckError"]:
        out["factCheckError"] = " | ".join(str(x) for x in fc["errors"])
    out["trueFacts"] = claims["supported"]
    out["falseFacts"] = claims["contradicted"]
    return out
