"""Retrain WELFake BiLSTM (leakage-safe 80/10/10). Usage: python python/train_lstm_welfake.py"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, Input, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WELFake_Dataset.csv"
MODELS = ROOT / "models"
MODEL_PATH = MODELS / "fake_news_model.h5"
TOKENIZER_PATH = MODELS / "tokenizer.pkl"

MAX_WORDS = 50000
MAX_LEN = 300
RANDOM_STATE = 42
TEST_SIZE = 0.10
VAL_SIZE_OF_REMAINING = 0.10 / 0.90
EPOCHS = 5
BATCH_SIZE = 64


def main() -> None:
    if not DATA_PATH.is_file():
        raise SystemExit(f"Немає датасету: {DATA_PATH}")
    MODELS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    df = df[df["text"].str.len() > 0]
    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    print(f"WELFake rows: {before} -> {len(df)} after exact-text dedupe")

    X = df["text"].values
    y = df["label"].values.astype(np.int32)

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
    print(
        f"Split train/val/test: {len(X_train)}/{len(X_val)}/{len(X_test)} "
        f"(~80/10/10 stratified rs={RANDOM_STATE})"
    )

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(list(X_train))

    def encode(texts):
        return pad_sequences(
            tokenizer.texts_to_sequences(list(texts)),
            maxlen=MAX_LEN,
            padding="pre",
            truncating="pre",
        )

    X_train_pad = encode(X_train)
    X_val_pad = encode(X_val)
    X_test_pad = encode(X_test)

    model = Sequential(
        [
            Input(shape=(MAX_LEN,)),
            Embedding(input_dim=MAX_WORDS, output_dim=128),
            Bidirectional(LSTM(64)),
            Dropout(0.6),
            Dense(32, activation="relu"),
            Dropout(0.4),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    history = model.fit(
        X_train_pad,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val_pad, y_val),
        verbose=1,
    )

    proba = model.predict(X_test_pad, batch_size=256, verbose=1).reshape(-1)
    pred = (proba >= 0.5).astype(np.int32)
    print("\n=== holdout test (tokenizer fit on train only; explicit val) ===")
    print(classification_report(y_test, pred, digits=4, zero_division=0))
    cm = confusion_matrix(y_test, pred)
    print("Confusion matrix:\n", cm)
    acc = float(accuracy_score(y_test, pred))
    f1 = float(f1_score(y_test, pred, average="macro", zero_division=0))

    model.save(MODEL_PATH)
    with TOKENIZER_PATH.open("wb") as f:
        pickle.dump(tokenizer, f)

    best_val = max(history.history.get("val_accuracy", [0.0]))
    lines = [
        "WELFake BiLSTM retrain metrics (leakage-safe)",
        "architecture: Embedding(50000,128) -> Bidirectional(LSTM(64)) -> Dropout(0.6) -> Dense(32) -> Dropout(0.4) -> Dense(1,sigmoid)",
        f"deduped_n={len(df)} removed={before - len(df)}",
        f"split=train/val/test ~80/10/10 stratified rs={RANDOM_STATE}",
        f"n_train={len(X_train)} n_val={len(X_val)} n_test={len(X_test)}",
        "tokenizer.fit_on_texts on TRAIN only; val/test transformed with frozen vocab",
        "validation_data=explicit holdout (no validation_split)",
        f"epochs={EPOCHS} batch={BATCH_SIZE}",
        f"best_val_accuracy={best_val:.4f}",
        f"test_accuracy={acc:.4f}",
        f"test_macro_f1={f1:.4f}",
        f"confusion_matrix={cm.tolist()}",
        "label_map: 0=real, 1=fake; sigmoid ≈ P(fake)",
    ]
    print("Збережено:", MODEL_PATH, TOKENIZER_PATH)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
