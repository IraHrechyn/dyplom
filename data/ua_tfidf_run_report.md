# Звіт запуску: український TF-IDF + LR

## Версія даних

- Файл: `data/ua_news_1000_1000.csv`
- Розмір: **14241477** байт
- Рядків у CSV (до очищення): **2000**
- Після dropna / min_len≥40: **2000**
- Після дедуплікації `body_text`: **2000**
- Розподіл міток (0=fake, 1=real): `{0: 1000, 1: 1000}`

## Середовище

- Python: `3.13.14` (Windows 10)
- scikit-learn: `1.8.0`
- pandas: `2.3.3`
- joblib: `1.5.3`

## Протокол (як у `python/train_tfidf_ua.py`)

- Split train/val/test ≈ **0.70 / 0.15 / 0.15**, `random_state=42`, stratify
- Фактичний split: **1400 / 300 / 300**
- TF-IDF: `max_features=15000`, `ngram_range=(1, 3)`, `min_df=2`, `sublinear_tf=True`
- LR: `C=8.0`, `class_weight=balanced`, `solver=lbfgs`
- Vectorizer + LR для holdout: fit на **train+val**, оцінка на **test** (production recipe)
- Примітка: артефакти models/*.joblib у звіті не перезаписувались; наявні файли: `fake_news_model.joblib` (120863 B), `tfidf_vectorizer.joblib` (812153 B)

## Holdout test

- Accuracy: **0.9200**
- Macro-F1: **0.9196**

### Матриця помилок

Рядки — істинний клас, стовпці — прогноз (`0=fake`, `1=real`):

```
[[127  23]
 [  1 149]]
```

- TN (0→0) = 127, FP (0→1) = 23
- FN (1→0) = 1, TP (1→1) = 149
- Accuracy check: (127 + 149) / 300 = **0.9200**

### Classification report

```
              precision    recall  f1-score   support

           0     0.9922    0.8467    0.9137       150
           1     0.8663    0.9933    0.9255       150

    accuracy                         0.9200       300
   macro avg     0.9292    0.9200    0.9196       300
weighted avg     0.9292    0.9200    0.9196       300
```

## Відтворення

1. Покладіть локально CSV у `data/ua_news_1000_1000.csv`.
2. `pip install -r requirements.txt` (потрібен `scikit-learn==1.8.0`, Python ≥ 3.11).
3. `python python/report_ua_tfidf.py` — оновить цей звіт, або `python python/train_tfidf_ua.py` — повне навчання.

