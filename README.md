# Fake News Detector

Вебзастосунок (Next.js) і Telegram-бот для оцінки ризику фейкових новин: стильова класифікація + LLM-фактчек.

## Вимоги

- Node.js **>= 20.9**
- Python **3.11+** (вимога `scikit-learn==1.8.0`)
- ключі API у `.env.local` (див. `.env.example`)

## Встановлення

```bash
npm install
pip install -r requirements.txt
cp .env.example .env.local
```

Заповніть `.env.local` своїми ключами. Файл із секретами не комітиться.

## Запуск вебверсії

```bash
npm run dev
```

Відкрийте http://localhost:3000

## Запуск Telegram-бота

Спочатку запустіть вебсервер (`npm run dev`), потім:

```bash
python bot/telegram_bot.py
```

Бот звертається до `FACTCHECK_API_URL` (за замовчуванням `http://127.0.0.1:3000/api/factcheck`).

## Моделі

Готові артефакти лежать у `models/`:

- `fake_news_model.joblib` + `tfidf_vectorizer.joblib` — український TF-IDF + LR (`0=fake`, `1=real`)
- `fake_news_model.h5` + `tokenizer.pkl` — англійський **Bidirectional LSTM** (WELFake): Embedding → BiLSTM(64) → Dropout(0.6) → Dense(32) → Dropout(0.4) → Dense(1); у CSV WELFake `0=real`, `1=fake` (sigmoid ≈ P(fake))

Маршрутизація: український текст → TF-IDF; англійський → LSTM. Soft vote: стиль **60%**, Gemini **20%**, OpenRouter **20%**.

## Дані для відтворення експерименту

Покладіть файли в `data/` (каталог у `.gitignore`):

- `ua_news_1000_1000.csv` — український корпус (`|`-роздільник; мітки `0=fake`, `1=real`)
- `WELFake_Dataset.csv` — англомовний WELFake

## Повторення навчання / оцінки

```bash
python python/train_tfidf_ua.py
python python/eval_lstm_welfake.py
python python/train_lstm_welfake.py
```

Скрипти спочатку прибирають точні дублікати текстів, ділять вибірку, і лише потім підганяють TF-IDF / tokenizer на **train**.

## Структура

- `app/` — Next.js UI та API
- `python/` — аналіз тексту, OCR, витяг статей, скрипти навчання
- `bot/` — Telegram-бот
- `models/` — навчені моделі

## Примітки

- Node.js ≥ 20.9; Python ≥ 3.11; на Windows Python-процеси запускаються з `PYTHONUTF8=1`.
- Завантаження URL блокує localhost / приватні IP та обмежує редиректи.
- `scikit-learn==1.8.0` зафіксовано під наявні `.joblib` (тому мінімум Python 3.11).
