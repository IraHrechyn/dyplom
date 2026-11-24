from __future__ import annotations

import logging
import os
import re
from typing import Any

import requests

log = logging.getLogger("factcheck_client")

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

def _factcheck_has_evidence(fc: dict[str, Any]) -> bool:
    o = fc.get("overall") or {}
    if not isinstance(o, dict):
        return False
    fake = o.get("fakePercent")
    truth = o.get("truthPercent")
    try:
        fake_f = float(fake)
        truth_f = float(truth)
    except (TypeError, ValueError):
        return False
    if not (0.0 <= fake_f <= 100.0 and 0.0 <= truth_f <= 100.0):
        return False
    if o.get("verdict") == "NOT_ENOUGH_INFO":
        return False
    if fc.get("plausibilityOnly"):
        return False
    sources = fc.get("sourcesUsed")
    if not isinstance(sources, list) or len(sources) == 0:
        return False
    for c in fc.get("claims") or []:
        if not isinstance(c, dict):
            continue
        if c.get("verdict") not in ("SUPPORTED", "CONTRADICTED"):
            continue
        citations = c.get("citations") if isinstance(c.get("citations"), list) else []
        if any(isinstance(x, dict) and str(x.get("url") or "").strip() for x in citations):
            return True
    return False


def soft_vote_combined(style_result: dict[str, Any], fact_checks: list[dict[str, Any]]) -> dict[str, Any]:
    ensemble_fake = float(style_result.get("fakePercentage") or 0)
    mb = style_result.get("modelBreakdown") or {}
    votes: list[tuple[str, float, float]] = []

    style_w = TRUST["style"]
    active = str(mb.get("activeModel") or "")
    sk_w = float(mb.get("sklearnWeight") or 0)
    ke_w = float(mb.get("kerasWeight") or 0)

    def _in_range_pct(n: Any) -> bool:
        try:
            v = float(n)
        except (TypeError, ValueError):
            return False
        return 0.0 <= v <= 100.0

    if not _in_range_pct(ensemble_fake):
        ensemble_fake = 50.0
    else:
        ensemble_fake = float(ensemble_fake)

    if active == "tfidf" or sk_w > ke_w:
        votes.append(("TF-IDF (UA)", ensemble_fake, style_w))
    elif active == "keras" or ke_w > 0 or mb:
        votes.append(("Keras (EN)", ensemble_fake, style_w))
    else:
        votes.append(("стиль", ensemble_fake, style_w))

    usable = []
    for fc in fact_checks:
        if not _factcheck_has_evidence(fc):
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
        votes.append(
            (
                str(fc.get("provider") or "LLM"),
                float(fc["overall"]["fakePercent"]),
                w,
            )
        )

    if not votes:
        return {
            "combinedFakePercent": ensemble_fake,
            "combinedTruthPercent": round(100.0 - ensemble_fake, 1),
            "note": "Фінальна оцінка за стилем тексту.",
        }

    sum_w = sum(w for _, _, w in votes) or 1.0
    combined_fake = round(sum(fake * w for _, fake, w in votes) / sum_w, 1)
    if not (0.0 <= combined_fake <= 100.0):
        combined_fake = ensemble_fake
    log.info(
        "[soft vote] моделі: %s",
        [
            {
                "model": name,
                "riskPercent": fake,
                "truthPercent": round(100.0 - fake, 1),
                "weight": w,
                "weightShare": f"{round((w / sum_w) * 1000) / 10}%",
            }
            for name, fake, w in votes
        ],
    )
    log.info(
        "[soft vote] підсумок: %s",
        {
            "riskPercent": combined_fake,
            "truthPercent": round(100.0 - combined_fake, 1),
            "activeStyleModel": (mb.get("activeModel") if isinstance(mb, dict) else None),
            "language": (mb.get("language") if isinstance(mb, dict) else None),
        },
    )
    if usable:
        note = "Фінальна оцінка за результатами повної перевірки."
    elif fact_checks:
        note = "Фактчек виконано; на діаграмі стильова оцінка (брак перевірених доказів не змінює ризик)."
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
    disputed_limit: int = 6,
    insufficient_limit: int = 6,
) -> dict[str, list[dict[str, Any]]]:
    supported: list[dict[str, Any]] = []
    contradicted: list[dict[str, Any]] = []
    insufficient: list[dict[str, Any]] = []

    for fc in fact_checks:
        provider = str(fc.get("provider") or "").strip() or None
        for c in fc.get("claims") or []:
            if not isinstance(c, dict):
                continue
            claim = str(c.get("claim") or "").strip()
            if not claim:
                continue
            verdict = c.get("verdict")
            citations = c.get("citations") if isinstance(c.get("citations"), list) else []
            verified = []
            for x in citations:
                if not isinstance(x, dict):
                    continue
                url = str(x.get("url") or "").strip()
                quote = str(x.get("quote") or "").strip()
                if not url or not quote:
                    continue
                scope = str(x.get("verificationScope") or "").strip()
                item_cit: dict[str, Any] = {"url": url, "quote": quote}
                if scope in ("search_snippet", "full_page"):
                    item_cit["verificationScope"] = scope
                verified.append(item_cit)
            conf = max(0.0, min(1.0, float(c.get("confidence") or 0)))
            item = {
                "claim": claim,
                "confidence": conf,
                "note": str(c.get("note") or "").strip() or None,
                "citations": verified,
                "provider": provider,
                "verdict": verdict,
            }
            if verdict == "SUPPORTED":
                if not verified:
                    continue
                supported.append(item)
            elif verdict == "CONTRADICTED":
                if not verified:
                    continue
                contradicted.append(item)
            elif verdict == "NOT_ENOUGH_INFO":
                insufficient.append(item)

    supported_d = _dedupe_claims(supported)
    contradicted_d = _dedupe_claims(contradicted)
    insufficient_d = _dedupe_claims(insufficient)

    disputed: list[dict[str, Any]] = []
    used_c: set[int] = set()
    supported_final: list[dict[str, Any]] = []
    for s in supported_d:
        idx = next(
            (
                i
                for i, x in enumerate(contradicted_d)
                if i not in used_c
                and _claims_near_duplicate(str(s.get("claim") or ""), str(x.get("claim") or ""))
            ),
            -1,
        )
        if idx >= 0:
            c = contradicted_d[idx]
            used_c.add(idx)
            s_claim = str(s.get("claim") or "")
            c_claim = str(c.get("claim") or "")
            disputed.append(
                {
                    "claim": s_claim if len(s_claim) >= len(c_claim) else c_claim,
                    "verdict": "DISPUTED",
                    "confidence": max(float(s.get("confidence") or 0), float(c.get("confidence") or 0)),
                    "citations": (list(s.get("citations") or []) + list(c.get("citations") or []))[:4],
                    "sideVerdicts": [
                        {
                            "provider": s.get("provider"),
                            "verdict": "SUPPORTED",
                            "confidence": s.get("confidence"),
                            "note": s.get("note"),
                        },
                        {
                            "provider": c.get("provider"),
                            "verdict": "CONTRADICTED",
                            "confidence": c.get("confidence"),
                            "note": c.get("note"),
                        },
                    ],
                }
            )
        else:
            supported_final.append(s)

    contradicted_final = [c for i, c in enumerate(contradicted_d) if i not in used_c]
    taken = supported_final + contradicted_final + disputed
    insufficient_final: list[dict[str, Any]] = []
    for n in insufficient_d:
        if any(_claims_near_duplicate(str(n.get("claim") or ""), str(x.get("claim") or "")) for x in taken):
            continue
        insufficient_final.append(n)

    return {
        "supported": supported_final[:true_limit],
        "contradicted": contradicted_final[:false_limit],
        "disputed": disputed[:disputed_limit],
        "insufficient": insufficient_final[:insufficient_limit],
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
        timeout=180.0,
        fast=True,
        providers=["openrouter", "gemini"],
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
    out["disputedFacts"] = claims["disputed"]
    out["insufficientFacts"] = claims["insufficient"]
    return out
