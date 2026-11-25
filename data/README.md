# Datasets

| File | Purpose |
|------|---------|
| `ua_news_1000_1000.csv` | UA TF-IDF (роздільник `\|`; мітки `0=fake`, `1=real`; у git) |
| `WELFake_Dataset.csv` | English BiLSTM (мітки `0=real`, `1=fake`; не в git — покладіть локально) |

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
