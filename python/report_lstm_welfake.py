"""Write WELFake BiLSTM holdout run report (no model overwrite). Usage: python python/report_lstm_welfake.py"""

from __future__ import annotations

import pickle
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WELFake_Dataset.csv"
OUT_PATH = ROOT / "data" / "welfake_bilstm_run_report.md"
MODEL_PATH = ROOT / "models" / "fake_news_model.h5"
TOKENIZER_PATH = ROOT / "models" / "tokenizer.pkl"

MAX_LEN = 300
RANDOM_STATE = 42
TEST_SIZE = 0.10
VAL_SIZE_OF_REMAINING = 0.10 / 0.90


def main() -> None:
    if not DATA_PATH.is_file():
        raise SystemExit(f"Немає датасету: {DATA_PATH}")
    if not MODEL_PATH.is_file() or not TOKENIZER_PATH.is_file():
        raise SystemExit("Потрібні models/fake_news_model.h5 і models/tokenizer.pkl")

    raw_bytes = DATA_PATH.stat().st_size

    df0 = pd.read_csv(DATA_PATH)
    n0 = len(df0)
    df1 = df0.dropna(subset=["text", "label"]).copy()
    n1 = len(df1)
    df1["text"] = df1["text"].astype(str).str.strip()
    df1["label"] = df1["label"].astype(int)
    df2 = df1[df1["text"].str.len() > 0].copy()
    n2 = len(df2)
    counts_blank = {int(k): int(v) for k, v in df2["label"].value_counts().sort_index().items()}
    df = df2.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    n3 = len(df)
    counts = {int(k): int(v) for k, v in df["label"].value_counts().sort_index().items()}

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

    with TOKENIZER_PATH.open("rb") as f:
        tokenizer = pickle.load(f)
    model = load_model(MODEL_PATH)
    seq = tokenizer.texts_to_sequences(list(X_test))
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="pre", truncating="pre")
    proba = model.predict(padded, batch_size=256, verbose=1).reshape(-1)
    pred = (proba >= 0.5).astype(np.int32)
    cm = confusion_matrix(y_test, pred)
    acc = float(accuracy_score(y_test, pred))
    f1 = float(f1_score(y_test, pred, average="macro", zero_division=0))
    report = classification_report(y_test, pred, digits=4, zero_division=0)

    try:
        import tensorflow as tf

        tf_ver = tf.__version__
    except Exception:
        tf_ver = "unknown"

    lines = [
        "# Звіт запуску: WELFake BiLSTM",
        "",
        "Holdout-оцінка артефактів `models/fake_news_model.h5` + `models/tokenizer.pkl`.",
        "",
        "## Версія даних",
        "",
        f"- Файл: `data/{DATA_PATH.name}`",
        f"- Розмір: **{raw_bytes}** байт",
        f"- Очищення: **{n0}** → **{n1}** (dropna) → **{n2}** (порожні після trim) → **{n3}** (дедуплікація)",
        f"- Розподіл після trim (0=real, 1=fake): `{counts_blank}`",
        f"- Розподіл після дедуплікації: `{counts}`",
        "",
        "## Середовище",
        "",
        f"- Python: `{sys.version.split()[0]}` ({platform.system()} {platform.release()})",
        f"- scikit-learn: `{sklearn.__version__}`",
        f"- pandas: `{pd.__version__}`",
        f"- tensorflow: `{tf_ver}`",
        "",
        "## Артефакти моделі",
        "",
        f"- `{MODEL_PATH.name}`: {MODEL_PATH.stat().st_size} B",
        f"- `{TOKENIZER_PATH.name}`: {TOKENIZER_PATH.stat().st_size} B",
        "- Архітектура: Embedding(50000,128) → Bidirectional(LSTM(64)) → Dropout(0.6) → Dense(32) → Dropout(0.4) → Dense(1, sigmoid)",
        "- Мітки: `0=real`, `1=fake`; sigmoid ≈ P(fake)",
        "",
        "## Протокол (як у `python/eval_lstm_welfake.py`)",
        "",
        f"- Split train/val/test ≈ **80/10/10**, `random_state={RANDOM_STATE}`, stratify",
        f"- Фактичний split: **{len(X_train)} / {len(X_val)} / {len(X_test)}**",
        f"- `MAX_LEN={MAX_LEN}`; tokenizer vocab fit на train (файл `tokenizer.pkl`)",
        "- Оцінка: load `.h5`, predict на holdout test (без перенавчання)",
        "",
        "## Holdout test",
        "",
        f"- Accuracy: **{acc:.4f}**",
        f"- Macro-F1: **{f1:.4f}**",
        "",
        "### Матриця помилок",
        "",
        "Рядки — істинний клас, стовпці — прогноз (`0=real`, `1=fake`):",
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
        "1. Завантажте WELFake CSV і покладіть у `data/WELFake_Dataset.csv`.",
        "2. `pip install -r requirements.txt` (Python ≥ 3.11).",
        "3. Переконайтесь, що в `models/` є `fake_news_model.h5` і `tokenizer.pkl`.",
        "4. `python python/report_lstm_welfake.py` — оновить цей звіт; або `python python/eval_lstm_welfake.py`.",
        "",
    ]
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", OUT_PATH)
    print(f"clean={n0}->{n1}->{n2}->{n3} test_n={len(X_test)} acc={acc:.4f} cm={cm.tolist()}")


if __name__ == "__main__":
    main()
