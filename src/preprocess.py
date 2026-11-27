"""Course financial-news preprocessing, adapted to local CSV inputs."""
import argparse
import re
from pathlib import Path
import pandas as pd
from domain_terms import COMPANY_SPECIFIC_PHRASES, TECHNICAL_WHITELIST

COMPANIES = {"3231": "緯創", "3017": "奇鋐"}
ALLOWED_POS_PREFIX = ("n", "v", "a")
ALLOWED_POS_EXACT = {"eng", "nz", "vn"}
BASE_STOPWORDS = ['10月', '11月', '12月', '1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '上半年', '下半年', '不過', '之後', '今天', '今年', '以來', '以及', '傳出', '內容', '內文', '公司', '其中', '分別', '原文', '原文標題', '原標', '台灣', '同時', '問題', '因此', '大型', '大家', '如果', '對於', '市場', '很多', '應該', '指出', '推動', '提供', '整理', '新聞', '明顯', '昨日', '最後', '標題', '機制', '此外', '歷史', '法人', '由於', '目前', '相關', '第一', '第一季', '第三季', '第二季', '第四季', '美國', '股價', '表示', '記者', '認為', '近期', '連結', '還是', '長期', '預計']

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ''
    text = str(text)
    text = text.replace('\n', ' ')
    text = re.sub('http\\S+', ' ', text)
    text = re.sub('\\[[^\\]]+\\]', ' ', text)
    text = re.sub('\\s+', ' ', text)
    return text.strip()

def count_mentions(text: str, keyword: str) -> int:
    if not text:
        return 0
    return text.count(keyword)

def is_main_subject_article(title: str, content: str, stock_name: str, min_mentions: int=2, lead_check_chars: int=150) -> bool:
    title = clean_text(title)
    content = clean_text(content)
    lead = content[:lead_check_chars]
    full_text = f'{title} {content}'.strip()
    if stock_name in title:
        return True
    if stock_name in lead:
        return True
    if count_mentions(full_text, stock_name) >= min_mentions:
        return True
    return False

def build_title_lead_text(title: str, content: str, title_weight: int=3, lead_weight: int=2, lead_chars: int=180) -> str:
    title = clean_text(title)
    content = clean_text(content)
    lead = content[:lead_chars]
    parts = []
    parts.extend([title] * title_weight)
    parts.extend([lead] * lead_weight)
    return ' '.join([p for p in parts if p])

def monpa_tokenize_basic(text: str):
    text = clean_text(text)
    if not text:
        return []
    tokens = monpa.cut(text)
    filtered = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if len(tok) < 2:
            continue
        if re.fullmatch('[0-9０-９]+', tok):
            continue
        if re.fullmatch('[A-Za-z0-9\\-\\._:/]{8,}', tok):
            continue
        if tok in custom_stopwords:
            continue
        filtered.append(tok)
    return filtered

def get_jieba_pos_for_token(token: str):
    pairs = list(pseg.cut(token))
    if len(pairs) == 0:
        return None
    _, flag = pairs[0]
    return flag

def monpa_then_jieba_pos_tokenize(text: str):
    monpa_tokens = monpa_tokenize_basic(text)
    final_tokens = []
    for tok in monpa_tokens:
        flag = get_jieba_pos_for_token(tok)
        if tok in company_signal_terms:
            final_tokens.append(tok)
            continue
        if flag is None:
            continue
        if flag.startswith(ALLOWED_POS_PREFIX) or flag in ALLOWED_POS_EXACT:
            final_tokens.append(tok)
    return final_tokens

def monpa_then_jieba_pos_joined(text: str):
    return ' '.join(monpa_then_jieba_pos_tokenize(text))

def company_signal_filter(text: str, company_terms: set, min_ratio: float=0.02, min_count: int=2):
    text = clean_text(text)
    if not text:
        return (False, 0, 0.0)
    total_len = max(len(text), 1)
    hit_count = 0
    for term in company_terms:
        hit_count += text.count(term)
    hit_ratio = hit_count / total_len
    keep = hit_count >= min_count or hit_ratio >= min_ratio
    return (keep, hit_count, hit_ratio)

def preprocess(articles, stock_id, text_mode="title_lead"):
    # Lazy import allows text utility tests without the optional Chinese NLP stack.
    global monpa, jieba, pseg, company_signal_terms, custom_stopwords
    import monpa
    import jieba
    import jieba.posseg as pseg
    name = COMPANIES[stock_id]
    key = f"{stock_id}_{name}"
    company_signal_terms = set(COMPANY_SPECIFIC_PHRASES[key]) | set(TECHNICAL_WHITELIST[key]) | {name}
    custom_stopwords = set(BASE_STOPWORDS) | {name}
    for term in sorted(company_signal_terms):
        jieba.add_word(term)
    articles = articles.copy()
    articles["post_time"] = pd.to_datetime(articles["post_time"])
    articles = articles[articles["s_name"].isin(["Yahoo新聞", "Yahoo股市"])].copy()
    if "content_type" in articles:
        articles = articles[articles.content_type.eq("main") | articles.content_type.isna()].copy()
    articles = articles.sort_values("post_time")
    articles["trade_date"] = articles.post_time.dt.normalize()
    keep = articles.apply(lambda r: is_main_subject_article(r.title, r.content, name), axis=1)
    articles = articles[keep].copy()
    articles["weighted_text"] = articles.apply(lambda r: build_title_lead_text(r.title, r.content) + (" " + clean_text(r.content) if text_mode == "title_lead_full" else ""), axis=1)
    signal = articles.weighted_text.map(lambda t: company_signal_filter(t, company_signal_terms))
    articles["company_signal_count"] = signal.map(lambda x: x[1])
    articles["company_signal_ratio"] = signal.map(lambda x: x[2])
    articles = articles[signal.map(lambda x: x[0])].copy()
    articles["seg_text_concat"] = articles.weighted_text.map(monpa_then_jieba_pos_joined)
    return articles.groupby("trade_date").agg(
        article_count=("trade_date", "count"),
        weighted_text_concat=("weighted_text", lambda x: " ".join(x.astype(str))),
        seg_text_concat=("seg_text_concat", lambda x: " ".join(x.astype(str))),
        company_signal_count_sum=("company_signal_count", "sum"),
        company_signal_ratio_mean=("company_signal_ratio", "mean")
    ).reset_index()

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stock", choices=COMPANIES, required=True)
    p.add_argument("--articles", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--text-mode", choices=["title_lead", "title_lead_full"], default="title_lead")
    args = p.parse_args()
    result = preprocess(pd.read_csv(args.articles), args.stock, args.text_mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(result)} daily records")
