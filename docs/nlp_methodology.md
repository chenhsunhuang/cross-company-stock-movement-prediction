# Financial NLP and Statistical Vocabulary Selection

## News processing

The course data extraction selected Yahoo News and Yahoo Stock main articles mentioning the target company. The preprocessing workflow retains an article if the target company's name appears in the title or the first 150 characters of the body, or is mentioned at least twice in the title and body combined.

The primary representation repeats the title three times and the first 180 characters of the body twice. Company-specific phrases and technical terms are used to filter the resulting text: at least two matches or a match-count-to-character-length ratio of 0.02 is required. Matches may overlap. MONPA segments the weighted text; jieba part-of-speech tags retain nouns, verbs, adjectives and selected technical terms. Short tokens, standalone numbers, long alphanumeric strings and stopwords are removed. The pipeline aggregates article counts, weighted text, segmented text and company-signal measures by publication date.

The earlier text scope experiment compared weighted title and lead (3:2) against title, lead and full body (3:2:1). [Aggregate counts](../results/text_scope_counts.csv) summarize the eligible up/down dates and article counts for the 3-day, 1% target. The optional full-text mode in the portfolio script applies that representation to the refined preprocessing workflow, not an exact recreation of every earlier experiment setting.

## Directional vocabulary construction

All supervised statistics in the main LinearSVC experiment are estimated using training documents only. Candidate terms are obtained using CountVectorizer with minimum document frequency 3, maximum document frequency 0.8 and a unigram/bigram candidate range. The original binary-count vectorizer uses its default unigram analyzer; the archived document-frequency filter consequently favors supported unigrams. TF-IDF uses a unigram/bigram range. This behavior is retained for fidelity to the course implementation.

Let `p_up` and `p_down` be the class-specific fractions of training documents containing a term. The smoothed ratio is `(p_up + 1e-6) / (p_down + 1e-6)`. Up terms require a positive rate difference and ratio >= 2; down terms require a negative rate difference and ratio <= 1/2. Terms appearing in fewer than three documents or on the noise/company blacklists are excluded.

Within each directional pool, min-max normalize chi-square, mutual information, the relevant class's document rate, mean TF-IDF and directional ratio strength. The final score is:

`0.35 * chi2 + 0.25 * MI + 0.20 * document_rate + 0.10 * mean_TFIDF + 0.10 * ratio_strength`

The directional strength uses the ratio for up terms and its smoothed reciprocal for down terms. Up to 200 terms per pool are ranked and merged. The final TF-IDF vectorizer is fit on training text with this fixed vocabulary, then used to transform later text without refitting.

See [preprocessing](../src/preprocess.py), [domain dictionaries](../src/domain_terms.py) and [temporal grid](../src/temporal_grid.py).
