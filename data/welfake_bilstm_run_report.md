# Звіт запуску: WELFake BiLSTM

Holdout-оцінка артефактів `models/fake_news_model.h5` + `models/tokenizer.pkl`.

## Версія даних

- Файл: `data/WELFake_Dataset.csv`
- Розмір: **245086152** байт
- Очищення: **72134** → **72095** (dropna) → **71351** (порожні після trim) → **62704** (дедуплікація)
- Розподіл після trim (0=real, 1=fake): `{0: 35027, 1: 36324}`
- Розподіл після дедуплікації: `{0: 34619, 1: 28085}`

## Середовище

- Python: `3.13.14` (Windows 10)
- scikit-learn: `1.8.0`
- pandas: `2.3.3`
- tensorflow: `2.21.0`

## Артефакти моделі

- `fake_news_model.h5`: 78086216 B
- `tokenizer.pkl`: 13361297 B
- Архітектура: Embedding(50000,128) → Bidirectional(LSTM(64)) → Dropout(0.6) → Dense(32) → Dropout(0.4) → Dense(1, sigmoid)
- Мітки: `0=real`, `1=fake`; sigmoid ≈ P(fake)

## Протокол (як у `python/eval_lstm_welfake.py`)

- Split train/val/test ≈ **80/10/10**, `random_state=42`, stratify
- Фактичний split: **50162 / 6271 / 6271**
- `MAX_LEN=300`; tokenizer vocab fit на train (файл `tokenizer.pkl`)
- Оцінка: load `.h5`, predict на holdout test (без перенавчання)

## Holdout test

- Accuracy: **0.9490**
- Macro-F1: **0.9484**

### Матриця помилок

Рядки — істинний клас, стовпці — прогноз (`0=real`, `1=fake`):

```
[[3311  151]
 [ 169 2640]]
```

- TN (0→0) = 3311, FP (0→1) = 151
- FN (1→0) = 169, TP (1→1) = 2640
- Accuracy check: (3311 + 2640) / 6271 = **0.9490**

### Classification report

```
              precision    recall  f1-score   support

           0     0.9514    0.9564    0.9539      3462
           1     0.9459    0.9398    0.9429      2809

    accuracy                         0.9490      6271
   macro avg     0.9487    0.9481    0.9484      6271
weighted avg     0.9490    0.9490    0.9490      6271
```

## Відтворення

1. Завантажте WELFake CSV і покладіть у `data/WELFake_Dataset.csv`.
2. `pip install -r requirements.txt` (Python ≥ 3.11).
3. Переконайтесь, що в `models/` є `fake_news_model.h5` і `tokenizer.pkl`.
4. `python python/report_lstm_welfake.py` — оновить цей звіт; або `python python/eval_lstm_welfake.py`.

