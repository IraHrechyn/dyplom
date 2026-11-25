# Datasets

| File | Purpose |
|------|---------|
| `ua_news_1000_1000.csv` | UA TF-IDF (`\|` separator; 
`0=fake`, `1=real`) |
| `WELFake_Dataset.csv` | English BiLSTM (`0=real`, 
`1=fake` in this dump) |

## Звіти запуску

| Звіт | Експеримент |
|------|-------------|
| [`ua_tfidf_run_report.md`](./ua_tfidf_run_report.md) | UA TF-IDF + LR: holdout, матриця помилок, обсяги даних |
| [`welfake_bilstm_run_report.md`](./welfake_bilstm_run_report.md) | WELFake BiLSTM: holdout, матриця помилок, обсяги даних |

Оновити звіти (без перезапису моделей):

```bash
python python/report_ua_tfidf.py
python python/report_lstm_welfake.py
```

## Навчання / оцінка

```bash
python python/train_tfidf_ua.py
python python/eval_lstm_welfake.py
python python/train_lstm_welfake.py
```
