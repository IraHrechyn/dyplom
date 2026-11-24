"""Write UA TF-IDF holdout run report (no model overwrite). Usage: python python/report_ua_tfidf.py"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ua_news_1000_1000.csv"
OUT_PATH = ROOT / "data" / "ua_tfidf_run_report.md"

MAX_FEATURES = 15000
NGRAM_RANGE = (1, 3)
MIN_DF = 2
SUBLINEAR_TF = True
C = 8.0
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15


def _body(df: pd.DataFrame) -> pd.Series:
    title = df.get("title", pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
    text = df["text"].astype(str).str.strip()
    if "body" in df.columns:
        body = df["body"].fillna("").astype(str).str.strip()
        use_body = body.str.len() > 0
        merged = title + ". " + text
        merged = merged.where(~use_body, body)
        return merged.str.strip(". ").str.strip()
    return (title + ". " + text).str.strip(". ").str.strip()


def main() -> None:
    if not DATA_PATH.is_file():
        raise SystemExit(f"Немає датасету: {DATA_PATH}")

    raw_bytes = DATA_PATH.stat().st_size

    df = pd.read_csv(DATA_PATH, sep="|", engine="python", on_bad_lines="warn")
    n_raw = len(df)
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    df["body_text"] = _body(df)
    df = df[df["body_text"].str.len() >= 40].copy()
    before = len(df)
    df = df.drop_duplicates(subset=["body_text"], keep="first").reset_index(drop=True)
    counts = df["label"].value_counts().sort_index()

    X = df["body_text"].values
    y = df["label"].values
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=RANDOM_STATE, stratify=y_temp
    )

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        sublinear_tf=SUBLINEAR_TF,
        lowercase=True,
    )
    X_tv = vectorizer.fit_transform(np.concatenate([X_train, X_val]))
    y_tv = np.concatenate([y_train, y_val])
    model = LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=3000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    model.fit(X_tv, y_tv)
    pred = model.predict(vectorizer.transform(X_test))
    cm = confusion_matrix(y_test, pred)
    acc = float(accuracy_score(y_test, pred))
    f1 = float(f1_score(y_test, pred, average="macro", zero_division=0))
    report = classification_report(y_test, pred, digits=4, zero_division=0)

    art_note = "артефакти models/*.joblib у звіті не перезаписувались"
    model_path = ROOT / "models" / "fake_news_model.joblib"
    vec_path = ROOT / "models" / "tfidf_vectorizer.joblib"
    if model_path.is_file() and vec_path.is_file():
        art_note += (
            f"; наявні файли: `{model_path.name}` ({model_path.stat().st_size} B), "
            f"`{vec_path.name}` ({vec_path.stat().st_size} B)"
        )

    lines = [
        "# Звіт запуску: український TF-IDF + LR",
        "",
        "## Версія даних",
        "",
        f"- Файл: `data/{DATA_PATH.name}`",
        f"- Розмір: **{raw_bytes}** байт",
        f"- Рядків у CSV (до очищення): **{n_raw}**",
        f"- Після dropna / min_len≥40: **{before}**",
        f"- Після дедуплікації `body_text`: **{len(df)}**",
        f"- Розподіл міток (0=fake, 1=real): `{ {int(k): int(v) for k, v in counts.items()} }`",
        "",
        "## Середовище",
        "",
        f"- Python: `{sys.version.split()[0]}` ({platform.system()} {platform.release()})",
        f"- scikit-learn: `{sklearn.__version__}`",
        f"- pandas: `{pd.__version__}`",
        f"- joblib: `{joblib.__version__}`",
        "",
        "## Протокол (як у `python/train_tfidf_ua.py`)",
        "",
        f"- Split train/val/test ≈ **0.70 / 0.15 / 0.15**, `random_state={RANDOM_STATE}`, stratify",
        f"- Фактичний split: **{len(X_train)} / {len(X_val)} / {len(X_test)}**",
        f"- TF-IDF: `max_features={MAX_FEATURES}`, `ngram_range={NGRAM_RANGE}`, `min_df={MIN_DF}`, `sublinear_tf={SUBLINEAR_TF}`",
        f"- LR: `C={C}`, `class_weight=balanced`, `solver=lbfgs`",
        "- Vectorizer + LR для holdout: fit на **train+val**, оцінка на **test** (production recipe)",
        f"- Примітка: {art_note}",
        "",
        "## Holdout test",
        "",
        f"- Accuracy: **{acc:.4f}**",
        f"- Macro-F1: **{f1:.4f}**",
        "",
        "### Матриця помилок",
        "",
        "Рядки — істинний клас, стовпці — прогноз (`0=fake`, `1=real`):",
        "",
        "```",
        f"{cm}",
        "```",
        "",
        f"- TN (0→0) = {cm[0][0]}, FP (0→1) = {cm[0][1]}",
        f"- FN (1→0) = {cm[1][0]}, TP (1→1) = {cm[1][1]}",
        f"- Accuracy check: ({cm[0][0]} + {cm[1][1]}) / {cm.sum()} = "
        f"**{(cm[0][0] + cm[1][1]) / cm.sum():.4f}**",
        "",
        "### Classification report",
        "",
        "```",
        report.rstrip(),
        "```",
        "",
        "## Відтворення",
        "",
        "1. Покладіть локально CSV у `data/ua_news_1000_1000.csv`.",
        "2. `pip install -r requirements.txt` (потрібен `scikit-learn==1.8.0`, Python ≥ 3.11).",
        "3. `python python/report_ua_tfidf.py` — оновить цей звіт, або `python python/train_tfidf_ua.py` — повне навчання.",
        "",
    ]
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", OUT_PATH)
    print(f"test_n={len(X_test)} acc={acc:.4f} cm={cm.tolist()}")


if __name__ == "__main__":
    main()
