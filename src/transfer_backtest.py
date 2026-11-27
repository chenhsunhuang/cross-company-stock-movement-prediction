"""Team extension: fixed source vocabulary, target-company rolling LinearSVC.

Illustrates the archived exploratory design, not a production trading backtest.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from temporal_grid import load_inputs


def backtest(prices, daily, vocabulary, n=5, sigma=.01, window=60):
    prices = prices.sort_values("trade_date").copy()
    ret = prices.closing_price.shift(-n) / prices.closing_price - 1
    prices["label"] = np.where(ret.isna(), np.nan, np.where(ret > sigma, 1, np.where(ret < -sigma, 0, -1)))
    merged = prices.merge(daily, on="trade_date", how="inner").sort_values("trade_date").reset_index(drop=True)
    if len(merged) <= window or not vocabulary:
        raise ValueError("Insufficient matched dates or empty source vocabulary")
    scores = []
    for i in range(window, len(merged)):
        train = merged.iloc[i-window:i]
        train = train[train.label.isin([0, 1])]
        test = merged.iloc[[i]]
        if train.label.nunique() < 2 or not isinstance(test.seg_text_concat.iloc[0], str) or not test.seg_text_concat.iloc[0].strip() or not test.label.iloc[0] in (0, 1):
            continue
        vectorizer = TfidfVectorizer(vocabulary=sorted(set(vocabulary)))
        X = vectorizer.fit_transform(train.seg_text_concat.fillna(""))
        X, y = RandomOverSampler(random_state=42).fit_resample(X, train.label.astype(int))
        model = LinearSVC(max_iter=10000, dual=False).fit(X, y)
        score = float(model.decision_function(vectorizer.transform(test.seg_text_concat))[0])
        scores.append((score, int(test.label.iloc[0])))
    rows = []
    total = len(merged)-window
    for threshold in (0, .2, .4, .6, .8, 1):
        kept = [(s, y) for s, y in scores if abs(s) >= threshold]
        correct = sum(int(s > 0) == y for s, y in kept)
        rows.append({"threshold": threshold, "candidate_dates": total, "evaluated_calls": len(kept), "correct_calls": correct, "coverage": len(kept)/total, "accuracy": correct/len(kept) if kept else np.nan})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stock", default="3693")
    for key in ("prices", "daily-text", "up-vocab", "down-vocab", "output"):
        p.add_argument("--"+key, type=Path, required=True)
    args = p.parse_args()
    prices, daily = load_inputs(args.prices, args.daily_text, args.stock)
    terms = pd.concat([pd.read_csv(args.up_vocab).term, pd.read_csv(args.down_vocab).term]).dropna().tolist()
    result = backtest(prices, daily, terms)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
