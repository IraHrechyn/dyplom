# Datasets (not committed)

Place local copies here before running training scripts:

| File | Purpose |
|------|---------|
| `ua_news_1000_1000.csv` | UA TF-IDF (`\|` separator; `0=fake`, `1=real`) |
| `WELFake_Dataset.csv` | English BiLSTM (`0=real`, `1=fake` in this dump) |

Then:

```bash
python python/train_tfidf_ua.py
python python/eval_lstm_welfake.py
python python/train_lstm_welfake.py
```
