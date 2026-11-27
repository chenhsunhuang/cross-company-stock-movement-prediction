"""Training-only vocabulary selection and chronological holdout grid.

The archived course model is preserved, with local CSV inputs and explicit arguments.
"""
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

BLACKLIST = set(['京元電', '今天', '今年', '內容', '前線', '創高', '原文', '名單', '國票', '大事', '大漲', '字頭', '手機', '指出', '排名', '收盤', '收紅', '收黑', '整理', '新聞', '昨日', '標題', '漲停', '漲幅', '申請', '相關', '破底', '突然', '翻紅', '表示', '計算', '記者', '認為', '跌停', '跌幅', '重挫', '韭菜', '高雄', '麗臺'])

def make_label(ret, sigma):
    if pd.isna(ret):
        return np.nan
    if ret > sigma:
        return 'up'
    elif ret < -sigma:
        return 'down'
    else:
        return 'neutral'

def minmax_scale(series):
    s = series.astype(float)
    s_min = s.min()
    s_max = s.max()
    if s_max == s_min:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s_min) / (s_max - s_min)

def build_vocab_with_tables(vocab_stats_df, top_k=200, min_doc_count=3, ratio_threshold=2.0, blacklist_terms=None):
    df = vocab_stats_df.copy()
    df = df[(df['total_doc_count'] >= min_doc_count) & ((df['up_doc_count'] >= 1) | (df['down_doc_count'] >= 1))].copy()
    if blacklist_terms is not None and len(blacklist_terms) > 0:
        df = df[~df['term'].isin(blacklist_terms)].copy()
    eps = 1e-06
    df['ratio'] = (df['up_doc_rate'] + eps) / (df['down_doc_rate'] + eps)
    up_pool = df[(df['rate_diff'] > 0) & (df['ratio'] >= ratio_threshold)].copy()
    down_pool = df[(df['rate_diff'] < 0) & (df['ratio'] <= 1 / ratio_threshold)].copy()
    for pool, prefix in [(up_pool, 'up'), (down_pool, 'down')]:
        if not pool.empty:
            pool['chi2_scaled'] = minmax_scale(pool['chi2'])
            pool['mi_scaled'] = minmax_scale(pool['mi'])
            pool['doc_scaled'] = minmax_scale(pool[f'{prefix}_doc_rate'])
            pool['tfidf_scaled'] = minmax_scale(pool[f'{prefix}_tfidf_mean'])
            pool['ratio_strength'] = np.where(prefix == 'up', pool['ratio'], 1 / (pool['ratio'] + 1e-06))
            pool['ratio_scaled'] = minmax_scale(pool['ratio_strength'])
            pool['final_score'] = 0.35 * pool['chi2_scaled'] + 0.25 * pool['mi_scaled'] + 0.2 * pool['doc_scaled'] + 0.1 * pool['tfidf_scaled'] + 0.1 * pool['ratio_scaled']
    if up_pool.empty:
        up_vocab_df = pd.DataFrame(columns=df.columns.tolist() + ['chi2_scaled', 'mi_scaled', 'doc_scaled', 'tfidf_scaled', 'ratio_strength', 'ratio_scaled', 'final_score'])
    else:
        up_vocab_df = up_pool.sort_values(['final_score', 'ratio', 'up_doc_rate'], ascending=[False, False, False]).head(top_k).copy()
    if down_pool.empty:
        down_vocab_df = pd.DataFrame(columns=df.columns.tolist() + ['chi2_scaled', 'mi_scaled', 'doc_scaled', 'tfidf_scaled', 'ratio_strength', 'ratio_scaled', 'final_score'])
    else:
        down_vocab_df = down_pool.sort_values(['final_score', 'ratio', 'down_doc_rate'], ascending=[False, True, False]).head(top_k).copy()
    combined_vocab = sorted(list(set(up_vocab_df['term']).union(set(down_vocab_df['term']))))
    return (up_vocab_df, down_vocab_df, pd.DataFrame({'term': combined_vocab}), combined_vocab)

def run_one_nsigma(daily_df, daily_text_features_df, n, sigma, train_ratio=0.8, top_k=50, min_doc_count=3, ratio_threshold=2.0, blacklist_terms=None, models=None, save_vocab_csv=True, vocab_output_dir='fast_vocab_outputs', stock_id='', stock_name='', company_blacklist_terms=None):
    if models is None:
        models = {'LinearSVC': LinearSVC(max_iter=10000)}
    work_df = daily_df.copy().sort_values('trade_date')
    work_df[f'future_price_{n}d'] = work_df['closing_price'].shift(-n)
    work_df[f'future_return_{n}d'] = work_df[f'future_price_{n}d'] / work_df['closing_price'] - 1
    label_col = f"label_{n}d_sigma_{str(sigma).replace('.', '_')}"
    y_col = f"y_{n}d_sigma_{str(sigma).replace('.', '_')}"
    work_df[label_col] = work_df[f'future_return_{n}d'].apply(lambda x: make_label(x, sigma))
    model_df = work_df[work_df[label_col].isin(['up', 'down'])].copy()
    model_df[y_col] = model_df[label_col].map({'down': 0, 'up': 1})
    final_df = pd.merge(model_df, daily_text_features_df, on='trade_date', how='left').dropna(subset=['seg_text_concat'])
    final_df = final_df[final_df['article_count'] > 0].copy()
    final_df = final_df.sort_values('trade_date').reset_index(drop=True)
    if len(final_df) < 30:
        return (pd.DataFrame([{'n': n, 'sigma': sigma, 'model': 'NA', 'sample_size': len(final_df), 'accuracy': np.nan, 'f1_up': np.nan, 'note': 'sample too small'}]), None, None, None)
    split = int(len(final_df) * train_ratio)
    train_df = final_df.iloc[:split].copy()
    test_df = final_df.iloc[split:].copy()
    if len(train_df) < 20 or len(test_df) < 5:
        return (pd.DataFrame([{'n': n, 'sigma': sigma, 'model': 'NA', 'sample_size': len(final_df), 'accuracy': np.nan, 'f1_up': np.nan, 'note': 'train/test too small'}]), None, None, None)
    X_text_train = train_df['seg_text_concat']
    X_text_test = test_df['seg_text_concat']
    y_train = train_df[y_col]
    y_test = test_df[y_col]
    cv = CountVectorizer(token_pattern='(?u)\\b\\w+\\b', min_df=3, max_df=0.8, ngram_range=(1, 2))
    try:
        X_count_train = cv.fit_transform(X_text_train)
    except ValueError as exc:
        if "empty vocabulary" not in str(exc) and "no terms remain" not in str(exc).lower() and "max_df" not in str(exc):
            raise
        return pd.DataFrame([{"n": n, "sigma": sigma, "model": "NA", "accuracy": np.nan, "f1_up": np.nan, "note": "no candidate vocabulary"}]), None, None, None
    f_names = cv.get_feature_names_out()
    X_bin_train = CountVectorizer(vocabulary=f_names, binary=True).fit_transform(X_text_train)
    X_tf_train = TfidfVectorizer(vocabulary=f_names, ngram_range=(1, 2)).fit_transform(X_text_train)
    up_m = (y_train == 1).values
    dn_m = (y_train == 0).values
    if up_m.sum() == 0 or dn_m.sum() == 0:
        return (pd.DataFrame([{'n': n, 'sigma': sigma, 'model': 'NA', 'note': 'one class missing'}]), None, None, None)
    v_stats = pd.DataFrame({'term': f_names, 'up_doc_rate': np.asarray(X_bin_train[up_m].sum(axis=0)).ravel() / up_m.sum(), 'down_doc_rate': np.asarray(X_bin_train[dn_m].sum(axis=0)).ravel() / dn_m.sum(), 'chi2': chi2(X_bin_train, y_train)[0], 'mi': mutual_info_classif(X_bin_train, y_train, discrete_features=True, random_state=42), 'up_tfidf_mean': np.asarray(X_tf_train[up_m].mean(axis=0)).ravel(), 'down_tfidf_mean': np.asarray(X_tf_train[dn_m].mean(axis=0)).ravel(), 'total_doc_count': np.asarray(X_bin_train.sum(axis=0)).ravel(), 'up_doc_count': np.asarray(X_bin_train[up_m].sum(axis=0)).ravel(), 'down_doc_count': np.asarray(X_bin_train[dn_m].sum(axis=0)).ravel()})
    v_stats['rate_diff'] = v_stats['up_doc_rate'] - v_stats['down_doc_rate']
    effective_blacklist = set()
    if blacklist_terms is not None:
        effective_blacklist |= set(blacklist_terms)
    effective_blacklist |= set(company_blacklist_terms or ())
    up_v, dn_v, comb_v_df, comb_v = build_vocab_with_tables(v_stats, top_k=top_k, min_doc_count=min_doc_count, ratio_threshold=ratio_threshold, blacklist_terms=effective_blacklist)
    if len(comb_v) == 0:
        return (pd.DataFrame([{'n': n, 'sigma': sigma, 'model': 'NA', 'sample_size': len(final_df), 'train_size': len(train_df), 'test_size': len(test_df), 'accuracy': np.nan, 'f1_up': np.nan, 'note': 'combined vocab empty after filtering'}]), up_v, dn_v, comb_v_df)
    tfidf = TfidfVectorizer(vocabulary=comb_v, ngram_range=(1, 2))
    X_train = tfidf.fit_transform(X_text_train)
    X_test = tfidf.transform(X_text_test)
    if save_vocab_csv:
        tag = str(sigma).replace('.', '_')
        up_v.to_csv(os.path.join(vocab_output_dir, f'up_vocab_{stock_id}_{stock_name}_n{n}_sigma{tag}.csv'), index=False, encoding='utf-8-sig')
        dn_v.to_csv(os.path.join(vocab_output_dir, f'down_vocab_{stock_id}_{stock_name}_n{n}_sigma{tag}.csv'), index=False, encoding='utf-8-sig')
        comb_v_df.to_csv(os.path.join(vocab_output_dir, f'combined_vocab_{stock_id}_{stock_name}_n{n}_sigma{tag}.csv'), index=False, encoding='utf-8-sig')
    rows = []
    for m_name, m in models.items():
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        rows.append({'n': n, 'sigma': sigma, 'model': m_name, 'sample_size': len(final_df), 'train_size': len(train_df), 'test_size': len(test_df), 'accuracy': accuracy_score(y_test, y_pred), 'f1_up': f1_score(y_test, y_pred, zero_division=0)})
    return (pd.DataFrame(rows), up_v, dn_v, comb_v_df)

def company_blacklist(companies, stock_id, stock_name):
    terms = set()
    aliases = {str(stock_id), stock_name}
    for _, row in companies.iterrows():
        if str(row.company_id).strip() == str(stock_id):
            continue
        for col in ("company_name", "short_name"):
            value = row[col]
            if pd.isna(value):
                continue
            name = str(value).strip()
            for term in (name, name.replace("股份有限公司", "").replace("控股", "").strip()):
                if len(term) >= 2 and term not in aliases:
                    terms.add(term)
    return terms

def load_inputs(prices_path, daily_path, stock):
    prices = pd.read_csv(prices_path, dtype={"company_id": str})
    if "company_id" in prices:
        prices = prices[prices.company_id == str(stock)].copy()
    prices["trade_date"] = pd.to_datetime(prices.trade_date).dt.normalize()
    prices["closing_price"] = pd.to_numeric(prices.closing_price, errors="raise")
    daily = pd.read_csv(daily_path)
    daily["trade_date"] = pd.to_datetime(daily.trade_date).dt.normalize()
    if prices.empty or daily.empty or prices.trade_date.duplicated().any() or daily.trade_date.duplicated().any():
        raise ValueError("Inputs must contain nonempty, unique dates for the selected stock.")
    if not np.isfinite(prices.closing_price).all() or (prices.closing_price <= 0).any():
        raise ValueError("Closing prices must be finite and positive.")
    return prices.sort_values("trade_date"), daily.sort_values("trade_date")

def grid(prices, daily, stock, name, company_terms, output=None):
    rows = []
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
    for n in (1, 3, 5):
        for sigma in (0, 0.01, 0.03):
            r, _, _, _ = run_one_nsigma(prices, daily, n, sigma,
                train_ratio=0.8, top_k=200, min_doc_count=3, ratio_threshold=2.0,
                blacklist_terms=BLACKLIST, company_blacklist_terms=company_terms,
                save_vocab_csv=output is not None, vocab_output_dir=str(output),
                stock_id=stock, stock_name=name)
            rows.append(r)
    return pd.concat(rows, ignore_index=True).sort_values(["accuracy", "f1_up"], ascending=False)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stock", choices=["3231", "3017"], required=True)
    p.add_argument("--prices", type=Path, required=True)
    p.add_argument("--daily-text", type=Path, required=True)
    p.add_argument("--companies", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    name = {"3231": "緯創", "3017": "奇鋐"}[args.stock]
    prices, daily = load_inputs(args.prices, args.daily_text, args.stock)
    terms = company_blacklist(pd.read_csv(args.companies, dtype={"company_id": str}), args.stock, name)
    results = grid(prices, daily, args.stock, name, terms, args.output / "vocabularies")
    results.to_csv(args.output / "grid_results.csv", index=False)
    print(results.to_string(index=False))
