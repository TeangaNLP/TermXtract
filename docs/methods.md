# Methods

`TermExtractor.METHODS` and `SupervisedTermExtractor.METHODS` map each method name to its implementation, so `list(TermExtractor.METHODS)` lists the available methods. Passing an unknown `method` raises a `ValueError` that lists them too. To choose between them, see [Which method should I use?](benchmarks.md)

## Unsupervised: `TermExtractor(method=...)`

| `method`           | Description                                         | Extra parameters                      | Tutorial |
|--------------------|-----------------------------------------------------|---------------------------------------|----------|
| `tfidf`            | Term frequency–inverse document frequency           |                                       | [TF-IDF](tutorials/tfidf_tutorial.md) |
| `ridf`             | Residual IDF                                        |                                       | [TF-IDF](tutorials/tfidf_tutorial.md#residual-idf) |
| `cvalue`           | C-Value (frequent terms not nested in longer ones)  |                                       | [C-Value](tutorials/cvalue_tutorial.md) |
| `ncvalue`          | NC-Value (C-Value re-ranked by context words)       | `window_size`, `top_terms`, `cvalue_weight` | [C-Value](tutorials/cvalue_tutorial.md#nc-value) |
| `basic`            | Basic                                               | `alpha`                               | [C-Value](tutorials/cvalue_tutorial.md#basic-and-combobasic) |
| `combobasic`       | ComboBasic                                          | `alpha`, `beta`                       | [C-Value](tutorials/cvalue_tutorial.md#basic-and-combobasic) |
| `domaincoherence`  | Domain coherence                                    | `window_size`, `seed_terms`, `context_words` | [C-Value](tutorials/cvalue_tutorial.md#domain-coherence) |
| `domainpertinence` | Domain pertinence                                   | `reference_corpus` (required), `tie_break` | [Reference corpus](tutorials/reference_corpus_tutorial.md#domain-pertinence) |
| `weirdness`        | Weirdness                                           | `reference_corpus` (required)         | [Reference corpus](tutorials/reference_corpus_tutorial.md#weirdness) |
| `relevance`        | Relevance                                           | `reference_corpus` (required)         | [Reference corpus](tutorials/reference_corpus_tutorial.md#relevance) |
| `pmi`, `llr`, `chi2`, `dice`, `tscore` | Unithood: PMI, log-likelihood ratio, χ², Dice or t-score of the words of multi-word terms | `aggregation` | [Unithood](tutorials/unithood_tutorial.md) |
| `rake`             | Rapid Automatic Keyword Extraction                  | `stoplist`, `phrase_delimiters` (required) | [Keywords](tutorials/keyword_tutorial.md#rake) |
| `yake`             | YAKE! (local features of each document's words)     | `window_size`, `dedup_threshold`, `corpus_aggregation` | [Keywords](tutorials/keyword_tutorial.md#yake) |
| `textrank`         | TextRank, SingleRank or PositionRank (PageRank over a word graph) | `variant`, `window_size`, `graph` | [Keywords](tutorials/keyword_tutorial.md#textrank) |
| `topicmodeling`    | LDA topic-model-based scoring                       | `num_topics`, `top_words`             | [Topic models](tutorials/topic_model_tutorial.md) |
| `nmf`              | NMF topic-model-based scoring                       | `n_topics`, `max_features`            | [Topic models](tutorials/topic_model_tutorial.md#nmf) |
| `embedrank`        | EmbedRank or MDERank (sentence embeddings; needs the `transformers` extra) | `variant`, `corpus_aggregation`, `model_name`, `encoder`, `diversity` | [Embeddings](tutorials/embedding_tutorial.md) |

All methods accept `n` (maximum n-gram length; by default 2 for the unithood methods, 3 for `yake`, `textrank` and `embedrank` and 1 for the others), `threshold` (minimum score), `stopwords` (default `"english"`, or the ISO 639-1 code of another language, such as `"fr"`), `min_frequency` (default 1), `word_pattern` (default `"alphanumeric"`, dropping words without a letter or digit), and, for Teanga input, `words_layer`, `lemma_layer`, `pos_pattern` and `pos_layer`. `rake` does not use `n`, `stopwords`, `min_frequency`, `word_pattern`, `words_layer` or `lemma_layer`; it splits candidates at its own `stoplist`.

Any method's ranking can be re-ranked by semantic relatedness to some seed terms with `rerank=AdaText(seeds=...)` or `rerank=SemReRank(seeds=...)` from `termxtract.rerank` (needs the `transformers` extra; see [Re-ranking](tutorials/embedding_tutorial.md#re-ranking-with-related-words)).

## Supervised: `SupervisedTermExtractor(method=...)`

| `method`               | Description                                              | Extra parameters                                         | Tutorial |
|------------------------|----------------------------------------------------------|----------------------------------------------------------|----------|
| `roger`                | Ranking function of association measures, learned by evolution (multi-word terms only) | `population_size`, `offspring_size`, `generations`, `runs`, `random_state`, `max_negatives`, `patience`, `n_jobs` | [Roger](tutorials/supervised/roger.md) |
| `features`             | Classifier over the scores of the unsupervised methods and other features, as in HAMLET | `classifier`, `threshold`, `reference_corpus`, `random_state`, `n_jobs`, `min_frequency`, `word_pattern` | [Features](tutorials/supervised/features.md) |
| `token-classification` | Fine-tuned Hugging Face token-classification model (needs the `transformers` extra) | `model_name`, `epochs`, `batch_size`, `learning_rate`, `max_length`, `language`, `random_state`, `patience`, `sentences_layer` | [Token classification](tutorials/supervised/token_classification.md) |
| `llm`                  | In-context learning with a large language model, prompted with similar training sentences and their terms (needs the `llm` extra, and `transformers` for `selection="knn"`) | `client` (required), `k`, `template`, `selection`, `encoder`, `random_state`, `n_jobs`, `sentences_layer` | [In-context learning](tutorials/supervised/llm.md) |

All supervised methods accept `n` (maximum n-gram length; by default 2 for `roger`, 4 for `features`, 6 for `token-classification` and 10 for `llm`), and `stopwords` and `pos_pattern`, which filter the extracted terms but not the training candidates (except for `features`, which filters both), and `words_layer`, `lemma_layer` and `pos_layer` for Teanga input.

`termxtract.augmentation.augment()` adds synthetic training sentences for any supervised method, written by a large language model from the training sentences or by replacing their terms with similar ones from Wikipedia, as in Banerjee (2025, ch. 5) (see [Synthetic training data](tutorials/supervised/augmentation.md)).

## Results

`extract()` returns an `ATEResults` object with:

- `results.terms`: one entry per document, `{"doc_id": ..., "terms": [{"term": str, "score": float}, ...]}`, with terms ranked from highest to lowest score. Every method uses this format.
- `results.offsets`: for Teanga input, one entry per document, `{"doc_id": ..., "offsets": [{"term": str, "offset": (start, end)}, ...]}`, giving the character offsets of every occurrence of each extracted term. It is `None` for plain-text input.
- `results.corpus`: the input corpus.

`termxtract.evaluate` scores a ranking against a list of gold terms (see [Evaluation](api/evaluation.md)).
