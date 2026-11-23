"""Evaluate fake_news_model.h5 on WELFake holdout. Usage: python python/eval_lstm_welfake.py"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WELFake_Dataset.csv"
MODELS = ROOT / "models"
MODEL_PATH = MODELS / "fake_news_model.h5"
TOKENIZER_PATH = MODELS / "tokenizer.pkl"

MAX_LEN = 300
RANDOM_STATE = 42
TEST_SIZE = 0.10
VAL_SIZE_OF_REMAINING = 0.10 / 0.90


def main() -> None:
    if not DATA_PATH.is_file():
        raise SystemExit(f"Немає датасету: {DATA_PATH}")
    if not MODEL_PATH.is_file() or not TOKENIZER_PATH.is_file():
        raise SystemExit("Потрібні models/fake_news_model.h5 і models/tokenizer.pkl")

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    df = df[df["text"].str.len() > 0]
    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    print(f"WELFake rows: {before} -> {len(df)} after exact-text dedupe")

    X = df["text"].values
    y = df["label"].values
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=VAL_SIZE_OF_REMAINING,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )
    print(f"Split train/val/test: {len(X_train)}/{len(X_val)}/{len(X_test)}")

    with TOKENIZER_PATH.open("rb") as f:
        tokenizer = pickle.load(f)
    model = load_model(MODEL_PATH)
    print("Architecture:")
    model.summary()

    seq = tokenizer.texts_to_sequences(list(X_test))
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="pre", truncating="pre")
    proba = model.predict(padded, batch_size=256, verbose=1).reshape(-1)
    pred = (proba >= 0.5).astype(np.int32)

    print("\n=== holdout test (deduped WELFake, current .h5) ===")
    print(classification_report(y_test, pred, digits=4, zero_division=0))
    cm = confusion_matrix(y_test, pred)
    print("Confusion matrix:\n", cm)
    acc = float(accuracy_score(y_test, pred))
    f1 = float(f1_score(y_test, pred, average="macro", zero_division=0))
    print(f"accuracy={acc:.4f} macro_f1={f1:.4f}")

    lines = [
        "WELFake BiLSTM holdout metrics (fake_news_model.h5)",
        "architecture: Embedding(50000,128) -> Bidirectional(LSTM(64)) -> Dropout(0.6) -> Dense(32) -> Dropout(0.4) -> Dense(1,sigmoid)",
        f"deduped_n={len(df)} removed={before - len(df)}",
        f"split=train/val/test ~80/10/10 stratified rs={RANDOM_STATE}",
        f"n_train={len(X_train)} n_val={len(X_val)} n_test={len(X_test)}",
        "tokenizer.pkl = vocab fit on train only (same recipe as train_lstm_welfake.py)",
        f"test_accuracy={acc:.4f}",
        f"test_macro_f1={f1:.4f}",
        f"confusion_matrix={cm.tolist()}",
        "label_map: 0=real, 1=fake; sigmoid ≈ P(fake)",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
