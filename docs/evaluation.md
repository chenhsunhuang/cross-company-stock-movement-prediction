# Evaluation Protocol

## Target definition and holdout

For trading day t and horizon n, the label is based on `close[t+n] / close[t] - 1`. A return above sigma is up, below negative sigma is down, and otherwise neutral. Labels are computed on the complete price calendar before merging with daily text. Only non-neutral observations with news are retained. The earlier 80% of the sorted eligible observations form the training set and the remaining 20% form the holdout set. Vocabulary scoring, final TF-IDF fitting and LinearSVC training use the training partition.

The archived grid considers n in {1, 3, 5} and sigma in {0, 0.01, 0.03}. Accuracy is the fraction of correct labels; F1 (up) treats up as the positive class. Different targets produce different eligible dates and sample sizes. Results are exploratory because configurations are compared on their holdout partitions; the split does not purge forward-label overlap at the boundary. News is aligned by publication date, not verified intraday availability relative to the closing price.

The [saved grid results](../results/temporal_grid_results.csv) retain the 18 original training-only vocabulary experiments. The five-classifier comparison and broader course backtests are team work; their random-split scores are not used as evidence of prospective performance here.

For a prospective evaluation, freeze the target definition and vocabulary-selection rules using a development period, enforce label-availability cutoffs at all split boundaries, align publication times to a defined prediction cutoff, and compare against simple baselines on a separate untouched test period. No new performance estimates are claimed in this portfolio.
