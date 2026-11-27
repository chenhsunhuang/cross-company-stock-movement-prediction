"""Synthetic checks of data transformations, not financial performance tests."""
import sys
import unittest
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from preprocess import build_title_lead_text, is_main_subject_article, company_signal_filter
from temporal_grid import run_one_nsigma, make_label, company_blacklist
from transfer_backtest import backtest

class PipelineTests(unittest.TestCase):
    def test_relevance_and_weights(self):
        self.assertTrue(is_main_subject_article('緯創', '新聞內容', '緯創'))
        self.assertFalse(is_main_subject_article('市場', '其他新聞', '緯創'))
        self.assertEqual(build_title_lead_text('title', 'lead tail', lead_chars=4), 'title title title lead lead')
        self.assertTrue(company_signal_filter('散熱 散熱', {'散熱'})[0])

    def test_label_threshold(self):
        self.assertEqual(make_label(.02, .01), 'up')
        self.assertEqual(make_label(-.02, .01), 'down')
        self.assertEqual(make_label(.01, .01), 'neutral')
        self.assertTrue(pd.isna(make_label(np.nan, .01)))

    def test_company_blacklist(self):
        companies = pd.DataFrame({'company_id':['3231','3017'], 'company_name':['緯創股份有限公司','奇鋐股份有限公司'], 'short_name':['緯創','奇鋐']})
        terms = company_blacklist(companies, '3231', '緯創')
        self.assertIn('奇鋐', terms)
        self.assertNotIn('緯創', terms)

    def fixtures(self):
        dates = pd.date_range('2024-01-01', periods=101, freq='B')
        # Alternating up/down returns with both classes in all partitions.
        prices = pd.DataFrame({'trade_date':dates,'closing_price':[100 if i%2==0 else 104 for i in range(101)]})
        daily = pd.DataFrame({'trade_date': dates, 'article_count':1, 'seg_text_concat':['bull growth' if i%2==0 else 'bear decline' for i in range(101)]})
        daily.loc[80:, 'seg_text_concat'] += ' holdoutonly'
        return prices, daily

    def test_training_only_vocabulary(self):
        prices, daily = self.fixtures()
        result, up, down, combined = run_one_nsigma(prices, daily, 1, .01, top_k=200, save_vocab_csv=False)
        self.assertEqual(int(result.train_size.iloc[0]),80)
        self.assertEqual(int(result.test_size.iloc[0]),20)
        self.assertGreater(len(combined),0)
        self.assertFalse(any('holdoutonly' in term for term in combined.term))
        self.assertTrue(0 <= result.accuracy.iloc[0] <= 1)

    def test_transfer_coverage(self):
        prices, daily = self.fixtures()
        rows=backtest(prices,daily,['bull','growth','bear','decline'],n=1,sigma=.01,window=60)
        self.assertEqual(len(rows),6)
        self.assertTrue(rows.evaluated_calls.is_monotonic_decreasing)
        self.assertTrue((rows.coverage<=1).all())

if __name__ == '__main__':
    unittest.main()
