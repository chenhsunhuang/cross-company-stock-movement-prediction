# Data Requirements

The project uses a course-provided MySQL dataset combining financial news, stock prices and company metadata. Source tables were `stock_text`, `stock_prices` and `companies`. Obtain the data through authorized course or licensed sources. The repository does not distribute the news corpus, original database, historical price files, or generated tokenized text and vocabularies.

Store local inputs under `data/local/` (ignored by Git). Export data using your own authorized database client. No database connection is necessary for the portfolio scripts.

| Local file | Required columns |
|---|---|
| `articles_3231.csv` / `articles_3017.csv` | `title`, `content`, `post_time`, `s_name`; optional `content_type` |
| `prices.csv` | `company_id` (optional for a single company), `trade_date`, `closing_price` |
| `companies.csv` | `company_id`, `company_name`, `short_name` |
| `daily_3231.csv` / `daily_3017.csv` | `trade_date`, `article_count`, `weighted_text_concat`, `seg_text_concat`, `company_signal_count_sum`, `company_signal_ratio_mean` |

Use one price observation per trading date and company, positive prices and a consistent price definition. Export the complete trading calendar, including enough later prices to construct the forward-return labels. The modeling stage requires only `trade_date`, `article_count`, `seg_text_concat` from the daily text file.

## Generate daily text (optional)

MONPA and jieba are only necessary for preprocessing raw Chinese news. Install these in a Python environment supported by MONPA (a Python 3.10 environment is recommended for the optional NLP stack):

```bash
python -m pip install -r requirements-nlp.txt
python src/preprocess.py --stock 3231 --articles data/local/articles_3231.csv --output data/local/daily_3231.csv
```

## Cross-company experiment

Use the target firm's prices and daily text and explicit up/down source vocabularies, each containing a `term` column:

```bash
python src/transfer_backtest.py --stock 3693 --prices data/local/prices.csv --daily-text data/local/daily_3693.csv --up-vocab data/local/source_up.csv --down-vocab data/local/source_down.csv --output outputs/transfer.csv
```

Archived aggregate outcomes are already included in `results/`. The historical course database is not available here for a fresh model fit.
