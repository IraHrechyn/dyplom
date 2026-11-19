from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from datetime import date, datetime
from pathlib import Path

VERIFICATION_PUBLICATION = "publication"
VERIFICATION_TODAY = "today"

PY_DIR = Path(__file__).resolve().parent
ROOT = PY_DIR.parent
MODELS = ROOT / "models"
MODEL_PATH = MODELS / "fake_news_model.h5"
TOKENIZER_PATH = MODELS / "tokenizer.pkl"
SKLEARN_MODEL_PATH = MODELS / "fake_news_model.joblib"
TFIDF_VECTORIZER_PATH = MODELS / "tfidf_vectorizer.joblib"
if str(PY_DIR) not in sys.path:
    sys.path.insert(0, str(PY_DIR))

def detect_text_language(text: str) -> str:
    cyr = len(re.findall(r"[А-Яа-яІіЇїЄєҐґ]", text or ""))
    lat = len(re.findall(r"[A-Za-z]", text or ""))
    if cyr >= lat:
        return "uk"
    return "en"

def _infer_maxlen(model) -> int:
    try:
        inp = model.input_shape
        if inp and len(inp) > 1 and inp[1] is not None:
            return int(inp[1])
    except Exception:
        pass
    for layer in model.layers:
        try:
            cfg = layer.get_config()
            bis = cfg.get("batch_input_shape")
            if bis and len(bis) > 1 and bis[1] is not None:
                return int(bis[1])
        except Exception:
            continue
    return 500

def _sigmoid_to_p_fake(p_out: float) -> float:
    p_out = max(0.0, min(1.0, float(p_out)))
    raw_is_p_true = os.environ.get("FAKE_NEWS_RAW_IS_P_TRUE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    raw_is_p_fake = os.environ.get("FAKE_NEWS_RAW_IS_P_FAKE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if raw_is_p_true and not raw_is_p_fake:
        return 1.0 - p_out
    return p_out

def _predict_fake_probability(model, padded) -> float:
    import numpy as np

    raw = model.predict(padded, verbose=0, batch_size=1)
    raw = np.asarray(raw)

    if raw.ndim == 0:
        return _sigmoid_to_p_fake(float(raw))
    if raw.shape[-1] == 1:
        return _sigmoid_to_p_fake(float(raw.flat[0]))
    if raw.shape[-1] == 2:
        fake_idx = int(os.environ.get("FAKE_NEWS_SOFTMAX_FAKE_INDEX", "1").strip() or "1")
        fake_idx = max(0, min(1, fake_idx))
        return float(raw[0, fake_idx])
    return float(np.max(raw))

def _predict_p_fake_batch(model, padded) -> list[float]:
    import numpy as np

    n = len(padded)
    bs = max(1, min(32, n))
    raw = np.asarray(model.predict(padded, verbose=0, batch_size=bs))

    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    out: list[float] = []
    if raw.shape[-1] == 1:
        for i in range(raw.shape[0]):
            out.append(_sigmoid_to_p_fake(float(raw[i, 0])))
        return out
    if raw.shape[-1] == 2:
        fake_idx = int(os.environ.get("FAKE_NEWS_SOFTMAX_FAKE_INDEX", "1").strip() or "1")
        fake_idx = max(0, min(1, fake_idx))
        for i in range(raw.shape[0]):
            out.append(float(raw[i, fake_idx]))
        return out
    for i in range(raw.shape[0]):
        out.append(float(np.max(raw[i])))
    return out

def _split_sentences(text: str) -> list[str]:
    t = re.sub(r"\s+", " ", (text or "").strip())
    if not t:
        return []
    parts = re.split(r"(?<=[.!?…])\s+", t)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) >= 2]

_UI_NOISE_RE = re.compile(
    r"^(читати ще|читати далі|підписатись|підписатися|поділитись|поділитися|"
    r"share|subscribe|read more|continue reading|advertisement|реклама)\b",
    re.I,
)
_ATTR_ONLY_RE = re.compile(
    r"^(про це |про що |як )?повідомля(є|ють)\b|"
    r"^за (даними|інформацією|повідомленням)\b|"
    r"^(this is reported by|according to)\b|"
    r"^джерело\s*:",
    re.I,
)

def _is_usable_sentence(sentence: str, *, min_len: int) -> bool:
    s = re.sub(r"\s+", " ", (sentence or "").strip())
    if len(s) < min_len:
        return False
    cleaned = _UI_NOISE_RE.sub("", s).strip(" .,—–-")
    cleaned = re.sub(r"^(читати ще\s*)+", "", cleaned, flags=re.I).strip(" .,—–-")
    if len(cleaned) < min_len:
        return False
    words = cleaned.split()
    if len(words) < 5:
        return False
    if len(words) <= 12 and _ATTR_ONLY_RE.search(cleaned):
        return False
    return True

def _clean_sentence_text(sentence: str) -> str:
    s = re.sub(r"\s+", " ", (sentence or "").strip())
    s = re.sub(r"^(читати ще\s*)+", "", s, flags=re.I).strip(" .,—–-")
    return s

def _fragment_dict(sentence: str, p_fake: float) -> dict:
    p = max(0.0, min(1.0, float(p_fake)))
    return {
        "text": _clean_sentence_text(sentence),
        "trustPercent": round((1.0 - p) * 100, 1),
        "riskPercent": round(p * 100, 1),
    }

def _sklearn_fake_class_index(model) -> int:
    classes = list(getattr(model, "classes_", [0, 1]))
    try:
        wanted = int(os.environ.get("SKLEARN_FAKE_CLASS_INDEX", "0").strip() or "0")
    except ValueError:
        wanted = 0
    if wanted in classes:
        return classes.index(wanted)
    return 0

def _predict_sklearn_p_fake(model, vectorizer, texts: list[str]) -> list[float]:
    if not texts:
        return []
    X = vectorizer.transform(texts)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        fake_i = _sklearn_fake_class_index(model)
        return [max(0.0, min(1.0, float(row[fake_i]))) for row in proba]
    preds = model.predict(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    fake_label = classes[_sklearn_fake_class_index(model)]
    return [1.0 if p == fake_label else 0.0 for p in preds]

def _pick_sentences(text: str, max_sentences_scan: int, min_sentence_len: int) -> list[str]:
    sents_all = _split_sentences(text)
    seen: set[str] = set()
    sents: list[str] = []
    for s in sents_all:
        if not _is_usable_sentence(s, min_len=min_sentence_len):
            continue
        cleaned = _clean_sentence_text(s)
        key = re.sub(r"\s+", " ", cleaned.lower())[:200]
        if key in seen:
            continue
        seen.add(key)
        sents.append(cleaned)
        if len(sents) >= max_sentences_scan:
            break
    if not sents and len(text) >= min_sentence_len:
        step = 200
        for i in range(0, min(len(text), step * max_sentences_scan), step):
            chunk = _clean_sentence_text(text[i : i + step])
            if _is_usable_sentence(chunk, min_len=min_sentence_len):
                sents.append(chunk)
            if len(sents) >= max_sentences_scan:
                break
    return sents


def _split_reliable_questionable(
    sents: list[str], scores: list[float], top_k: int
) -> tuple[list[dict], list[dict]]:
    if not sents:
        return [], []
    scored = sorted(zip(sents, scores), key=lambda x: x[1])
    if len(scored) == 1:
        s, p = scored[0]
        fr = _fragment_dict(s, p)
        return ([], [fr]) if p >= 0.5 else ([fr], [])
    half = max(1, len(scored) // 2)
    k = min(top_k, half)
    reliable = [_fragment_dict(s, p) for s, p in scored[:k]]
    questionable = [_fragment_dict(s, p) for s, p in scored[-k:][::-1]]
    if reliable and questionable:
        avg_rel = sum(x["riskPercent"] for x in reliable) / len(reliable)
        avg_q = sum(x["riskPercent"] for x in questionable) / len(questionable)
        if avg_rel > avg_q:
            reliable, questionable = questionable, reliable
    return reliable, questionable


def _analyze_full_text(
    text: str,
    *,
    max_sentences_scan: int = 64,
    min_sentence_len: int = 22,
    top_k: int = 3,
) -> tuple[float, str, list[dict], list[dict], dict]:
    from news_article_extractor import NewsExtractionError

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import joblib

    lang = detect_text_language(text)
    excerpt = text[:400] + ("…" if len(text) > 400 else "")
    sents = _pick_sentences(text, max_sentences_scan, min_sentence_len)

    if lang == "uk":
        if not SKLEARN_MODEL_PATH.is_file() or not TFIDF_VECTORIZER_PATH.is_file():
            raise NewsExtractionError("Не знайдено українську модель TF-IDF+LR")
        sklearn_model = joblib.load(SKLEARN_MODEL_PATH)
        vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
        p_fake = _predict_sklearn_p_fake(sklearn_model, vectorizer, [text])[0]
        p_batch = _predict_sklearn_p_fake(sklearn_model, vectorizer, sents) if sents else []
        reliable, questionable = _split_reliable_questionable(sents, p_batch, top_k)
        model_breakdown = {
            "language": "uk",
            "activeModel": "tfidf",
            "sklearnFakePercentage": round(p_fake * 100, 1),
            "sklearnTruthPercentage": round((1.0 - p_fake) * 100, 1),
            "kerasFakePercentage": None,
            "kerasTruthPercentage": None,
            "sklearnWeight": 100.0,
            "kerasWeight": 0.0,
        }
        return p_fake, excerpt, reliable, questionable, model_breakdown

    from tensorflow import keras
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    if not MODEL_PATH.is_file() or not TOKENIZER_PATH.is_file():
        raise NewsExtractionError("Не знайдено англійську модель LSTM/Keras")
    model = keras.models.load_model(str(MODEL_PATH), compile=False)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    if not hasattr(tokenizer, "texts_to_sequences"):
        raise NewsExtractionError("Пошкоджений tokenizer.pkl")

    maxlen = _infer_maxlen(model)
    seq_full = tokenizer.texts_to_sequences([text])
    pad_full = pad_sequences(seq_full, maxlen=maxlen, padding="pre", truncating="pre")
    p_fake = max(0.0, min(1.0, float(_predict_fake_probability(model, pad_full))))

    p_batch: list[float] = []
    if sents:
        seqs = tokenizer.texts_to_sequences(sents)
        padded = pad_sequences(seqs, maxlen=maxlen, padding="pre", truncating="pre")
        p_batch = _predict_p_fake_batch(model, padded)
    reliable, questionable = _split_reliable_questionable(sents, p_batch, top_k)
    model_breakdown = {
        "language": "en",
        "activeModel": "keras",
        "sklearnFakePercentage": None,
        "sklearnTruthPercentage": None,
        "kerasFakePercentage": round(p_fake * 100, 1),
        "kerasTruthPercentage": round((1.0 - p_fake) * 100, 1),
        "sklearnWeight": 0.0,
        "kerasWeight": 100.0,
    }
    return p_fake, excerpt, reliable, questionable, model_breakdown

def _parse_published_date(published_at: str | None) -> date | None:
    if not published_at or not str(published_at).strip():
        return None
    s = str(published_at).strip()[:10]
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None

def _percentages_from_p_fake(p_fake: float) -> tuple[float, float]:
    p_fake = max(0.0, min(1.0, float(p_fake)))
    fake_pct = round(p_fake * 100, 1)
    truth_pct = round((1.0 - p_fake) * 100, 1)
    s = truth_pct + fake_pct
    if abs(s - 100.0) > 0.2 and s > 0:
        truth_pct = round(truth_pct / s * 100, 1)
        fake_pct = round(100.0 - truth_pct, 1)
    return truth_pct, fake_pct

def _adjust_p_fake_for_mode(
    p_fake: float,
    verification_mode: str,
    published_at: str | None,
    *,
    input_kind: str = "url",
) -> tuple[float, str]:
    p = max(0.0, min(1.0, float(p_fake)))
    pub_d = _parse_published_date(published_at)
    today = date.today()
    days_old: int | None = None
    if pub_d is not None:
        days_old = (today - pub_d).days
        if days_old < 0:
            days_old = 0

    if verification_mode == VERIFICATION_PUBLICATION:
        if input_kind == "plain_text":
            p_adj = p * 0.93
            note = "Режим «момент публікації» для вставленого тексту: застосовано легке пом'якшення (дати джерела немає)."
        elif days_old is not None:
            relief = 0.38 * min(1.0, days_old / 520.0)
            p_adj = p * (1.0 - relief)
            note = (
                f"Корекція для режиму «момент публікації»: матеріалу ~{days_old} дн. від дати на сторінці; "
                "показник ризику зменшено, щоб краще відобразити довіру «на той час»."
            )
        else:
            p_adj = p * 0.88
            note = (
                "Корекція для режиму «момент публікації»: дату на сторінці не знайдено; "
                "застосовано помірне пом'якшення показника."
            )
    else:
        if input_kind == "plain_text":
            p_adj = min(0.97, p + 0.06)
            note = "Режим «на сьогодні» для вставленого тексту: додано невеликий штраф невизначеності без дати джерела."
        elif days_old is not None and days_old > 14:
            penalty = 0.30 * min(1.0, (days_old - 14) / 420.0)
            p_adj = min(0.97, p + penalty)
            note = (
                f"Корекція для режиму «на сьогодні»: матеріалу ~{days_old} дн.; "
                "додано штраф за можливу застарілість фактів порівняно з оцінкою стилю."
            )
        elif days_old is not None:
            p_adj = min(0.97, p + 0.04)
            note = "Режим «на сьогодні»: свіжий матеріал — дуже невеликий штраф до показника актуальності."
        else:
            p_adj = min(0.97, p + 0.11)
            note = (
                "Режим «на сьогодні» без дати публікації: додано штраф невизначеності для актуальності формулювань."
            )

    p_adj = max(0.03, min(0.97, p_adj))
    return p_adj, note

def _append_related_reading(result: dict, text: str, req_timeout: int) -> None:
    skip = (os.environ.get("SKIP_RELATED_ARTICLES") or "0").strip().lower()
    if skip in ("1", "true", "yes", "on"):
        result["relatedArticles"] = []
        result["readingSearchQuery"] = ""
        result["sources"] = []
        return

    from related_articles import fetch_related_reading_payload

    try:
        extra = fetch_related_reading_payload(
            text,
            timeout=float(min(max(req_timeout, 4), 6)),
        )
    except Exception:
        extra = {"relatedArticles": [], "readingSearchQuery": ""}
    result["relatedArticles"] = extra.get("relatedArticles") or []
    result["readingSearchQuery"] = (extra.get("readingSearchQuery") or "").strip()
    result["sources"] = []

def analyze_raw_text(
    text: str,
    verification_mode: str = VERIFICATION_TODAY,
) -> dict:
    from news_article_extractor import NewsExtractionError

    if verification_mode not in (VERIFICATION_PUBLICATION, VERIFICATION_TODAY):
        raise NewsExtractionError(
            f"Невідомий режим перевірки: {verification_mode!r}. Допустимо: "
            f"{VERIFICATION_PUBLICATION}, {VERIFICATION_TODAY}"
        )

    text = (text or "").strip()
    if len(text) < 15:
        raise NewsExtractionError(
            "Текст занадто короткий для аналізу (потрібно щонайменше 15 символів)."
        )

    p_fake, excerpt, reliable_fr, questionable_fr, model_breakdown = _analyze_full_text(text)
    base_truth, base_fake = _percentages_from_p_fake(p_fake)
    p_adj, mode_note = _adjust_p_fake_for_mode(p_fake, verification_mode, None, input_kind="plain_text")
    adj_truth, adj_fake = _percentages_from_p_fake(p_adj)
    today_iso = date.today().isoformat()

    if verification_mode == VERIFICATION_PUBLICATION:
        mode_title = "Вставлений текст (умовно: момент публікації)"
        mode_detail = (
            "Дати публікації немає — це не посилання на сторінку. Опорна дата умовна: "
            f"{today_iso}. Режим «момент публікації» тут лише для узгодження з вибором для URL."
        )
        caveat = (
            "Оскільки текст вставлено вручну, автоматичної дати джерела немає. "
            "Відсотки — це оцінка мовних ознак моделлю."
        )
    else:
        mode_title = "Вставлений текст (на сьогодні)"
        mode_detail = f"Опорна дата: {today_iso}. Факти з базами не звіряються."
        caveat = (
            "Текст не завантажувався з інтернету — оцінюються лише формулювання в полі вводу."
        )

    out = {
        "text": text,
        "images": [],
        "videos": [],
        "publishedAt": None,
        "inputKind": "plain_text",
        "verificationMode": verification_mode,
        "referenceDate": today_iso,
        "modeTitle": mode_title,
        "modeDetail": mode_detail,
        "baseTruthPercentage": base_truth,
        "baseFakePercentage": base_fake,
        "truthPercentage": base_truth,
        "fakePercentage": base_fake,
        "adjustedTruthPercentage": adj_truth,
        "adjustedFakePercentage": adj_fake,
        "scoreNote": mode_note,
        "modelBreakdown": model_breakdown,
        "reliableFragments": reliable_fr,
        "questionableFragments": questionable_fr,
        "trueParts": [
            f"{mode_title}: загальна оцінка тексту — надійність ~{base_truth}% / ризик ~{base_fake}%.",
            caveat,
        ],
        "fakeParts": [
            mode_note,
            "Оцінка стосується стилю формулювань, а не незалежної юридичної верифікації фактів.",
        ],
        "sources": [],
    }
    _append_related_reading(out, text, 12)
    return out

def analyze_url(
    url: str,
    timeout: int = 15,
    verification_mode: str = VERIFICATION_PUBLICATION,
) -> dict:
    from news_article_extractor import DEFAULT_TIMEOUT, NewsExtractionError, extract_news_article

    if verification_mode not in (VERIFICATION_PUBLICATION, VERIFICATION_TODAY):
        raise NewsExtractionError(
            f"Невідомий режим перевірки: {verification_mode!r}. Допустимо: "
            f"{VERIFICATION_PUBLICATION}, {VERIFICATION_TODAY}"
        )

    extracted = extract_news_article(url, timeout=timeout or DEFAULT_TIMEOUT)
    text = (extracted.get("text") or "").strip()
    if not text:
        raise NewsExtractionError("Після зчитування сторінки текст порожній — немає що аналізувати")

    p_fake, excerpt, reliable_fr, questionable_fr, model_breakdown = _analyze_full_text(text)
    base_truth, base_fake = _percentages_from_p_fake(p_fake)

    today_iso = date.today().isoformat()
    published_raw = extracted.get("publishedAt")
    published_at = published_raw if isinstance(published_raw, str) and published_raw.strip() else None

    p_adj, mode_note = _adjust_p_fake_for_mode(p_fake, verification_mode, published_at, input_kind="url")
    adj_truth, adj_fake = _percentages_from_p_fake(p_adj)

    if verification_mode == VERIFICATION_PUBLICATION:
        reference_date = published_at or today_iso
        mode_title = "На момент публікації"
        mode_detail = (
            f"Опорна дата для інтерпретації: {reference_date}"
            + (
                f" (дата зі сторінки: {published_at})"
                if published_at
                else " (дату публікації на сторінці не знайдено — використано сьогоднішню дату як умовну точку відліку)"
            )
        )
        caveat = (
            "Матеріал оцінюється так, ніби формулювання відповідали реальності на момент виходу. "
            "Якщо згодом закони чи події змінились, це не робить стару статтю «фейковою» в журналістському сенсі."
        )
    else:
        reference_date = today_iso
        mode_title = "На сьогоднішній день"
        mode_detail = f"Опорна дата (сьогодні): {reference_date}"
        caveat = (
            "У цьому режимі показано мовні ознаки тексту відносно сьогоднішнього дня. "
            "Факти (закони, курси, події) автоматично не звіряються з базами — старий, але колись правдивий матеріал може виглядати підозріло без ручного фактчекінгу."
        )

    out = {
        **extracted,
        "inputKind": "url",
        "verificationMode": verification_mode,
        "referenceDate": reference_date,
        "modeTitle": mode_title,
        "modeDetail": mode_detail,
        "baseTruthPercentage": base_truth,
        "baseFakePercentage": base_fake,
        "truthPercentage": base_truth,
        "fakePercentage": base_fake,
        "adjustedTruthPercentage": adj_truth,
        "adjustedFakePercentage": adj_fake,
        "scoreNote": mode_note,
        "modelBreakdown": model_breakdown,
        "reliableFragments": reliable_fr,
        "questionableFragments": questionable_fr,
        "trueParts": [
            f"{mode_title}: загальна оцінка тексту — надійність ~{base_truth}% / ризик ~{base_fake}%. {mode_detail}",
            mode_note,
            caveat,
        ],
        "fakeParts": [
            f"Короткий уривок із статті: «{excerpt}»",
            "Оцінка стосується стилю формулювань, а не незалежної юридичної верифікації фактів.",
        ],
        "sources": [],
    }
    _append_related_reading(out, text, timeout or 15)
    return out

def main() -> None:
    parser = argparse.ArgumentParser(description="Аналіз URL або вставленого тексту (stdin JSON)")
    parser.add_argument("url", nargs="?", default=None, help="Посилання на новину")
    parser.add_argument(
        "--stdin",
        action="store_true",
        help='JSON з stdin: {"text":"...","verificationMode":"today"|"publication"}',
    )
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument(
        "--mode",
        choices=(VERIFICATION_PUBLICATION, VERIFICATION_TODAY),
        default=VERIFICATION_PUBLICATION,
        help="publication — контекст дати виходу; today — підкреслити актуальність на сьогодні",
    )
    parser.add_argument("--json", action="store_true", help="Лише JSON у stdout")
    args = parser.parse_args()

    try:
        if args.stdin:
            payload = json.load(sys.stdin)
            if not isinstance(payload, dict):
                raise ValueError("stdin має бути JSON-об'єктом")
            raw = (payload.get("text") or "").strip()
            vm = (payload.get("verificationMode") or VERIFICATION_TODAY).strip().lower()
            if vm not in (VERIFICATION_PUBLICATION, VERIFICATION_TODAY):
                vm = VERIFICATION_TODAY
            data = analyze_raw_text(raw, verification_mode=vm)
        elif args.url:
            data = analyze_url(args.url, timeout=args.timeout, verification_mode=args.mode)
        else:
            raise ValueError("Вкажіть URL або --stdin з JSON {text, verificationMode}")
    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}, ensure_ascii=False))
        else:
            print(f"Помилка: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(data, ensure_ascii=False))
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
