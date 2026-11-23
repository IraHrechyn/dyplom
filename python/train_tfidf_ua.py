"""Train UA TF-IDF + LR. Usage: python python/train_tfidf_ua.py"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ua_news_1000_1000.csv"
MODELS = ROOT / "models"
MODEL_PATH = MODELS / "fake_news_model.joblib"
VECTORIZER_PATH = MODELS / "tfidf_vectorizer.joblib"

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
        raise SystemExit(
            f"Немає датасету: {DATA_PATH}\n"
            "Покладіть CSV у data/ua_news_1000_1000.csv і запустіть знову."
        )
    MODELS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH, sep="|", engine="python", on_bad_lines="warn")
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    df["body_text"] = _body(df)
    df = df[df["body_text"].str.len() >= 40].copy()

    before = len(df)
    df = df.drop_duplicates(subset=["body_text"], keep="first").reset_index(drop=True)
    print(f"Записів після очищення: {len(df)} (прибрано дублікатів: {before - len(df)})")
    print("Розподіл міток:\n", df["label"].value_counts().sort_index().to_string())

    X = df["body_text"].values
    y = df["label"].values

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=RANDOM_STATE, stratify=y_temp
    )
    print(f"Split train/val/test: {len(X_train)}/{len(X_val)}/{len(X_test)}")

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        sublinear_tf=SUBLINEAR_TF,
        lowercase=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=3000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_tfidf, y_train)

    def report(name: str, Xs, ys) -> dict[str, float]:
        pred = model.predict(Xs)
        metrics = {
            "accuracy": float(accuracy_score(ys, pred)),
            "precision": float(precision_score(ys, pred, average="macro", zero_division=0)),
            "recall": float(recall_score(ys, pred, average="macro", zero_division=0)),
            "f1": float(f1_score(ys, pred, average="macro", zero_division=0)),
        }
        print(f"\n=== {name} ===")
        print(classification_report(ys, pred, digits=4, zero_division=0))
        print("Confusion matrix:\n", confusion_matrix(ys, pred))
        print({k: round(v, 4) for k, v in metrics.items()})
        return metrics

    val_metrics = report("validation", X_val_tfidf, y_val)
    report("holdout test (train-only vectorizer)", X_test_tfidf, y_test)

    X_tv = vectorizer.fit_transform(np.concatenate([X_train, X_val]))
    y_tv = np.concatenate([y_train, y_val])
    final = LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=3000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    final.fit(X_tv, y_tv)
    X_test_final = vectorizer.transform(X_test)
    final_pred = final.predict(X_test_final)
    print("\n=== holdout test after refit on train+val (production recipe) ===")
    print(classification_report(y_test, final_pred, digits=4, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, final_pred))
    print("classes_:", list(final.classes_), "(0=fake, 1=real)")

    joblib.dump(final, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    lines = [
        "UA TF-IDF + LR holdout metrics",
        f"dataset={DATA_PATH.name}",
        f"deduped_n={len(df)}",
        f"split=train/val/test ~{1 - TEST_SIZE - VAL_SIZE:.2f}/{VAL_SIZE:.2f}/{TEST_SIZE:.2f} rs={RANDOM_STATE}",
        "vectorizer fit on train only during selection; production refit on train+val",
        f"max_features={MAX_FEATURES} ngram={NGRAM_RANGE} C={C}",
        f"test_accuracy={accuracy_score(y_test, final_pred):.4f}",
        f"test_macro_f1={f1_score(y_test, final_pred, average='macro', zero_division=0):.4f}",
        f"selection_val_f1={val_metrics['f1']:.4f}",
        "label_map: 0=fake, 1=real",
    ]
    print("Збережено:", MODEL_PATH)
    print("Збережено:", VECTORIZER_PATH)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
