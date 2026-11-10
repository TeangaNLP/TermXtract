# Changelog

All notable changes to TermXtract are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Until 1.0.0,
breaking changes may happen in minor releases (0.x.0) and are listed here under
**Changed** or **Removed**, marked *breaking*.

## [Unreleased]

### Added

- `termxtract.evaluate()`, which scores extracted terms against a gold standard: a set of
  terms, one set per document, or a Teanga corpus with a span layer of gold terms (#34).
  It reports precision, recall, F1 and Matthews correlation coefficient over the corpus
  or per document (micro- or macro-averaged), and average precision, R-precision and
  precision at K of the ranking.
- `termxtract.datasets`, which loads term-extraction datasets as Teanga corpora with
  their gold terms in a `terms` layer (#35). Datasets are downloaded from their original
  sources on first use, checked against a SHA-256 checksum, and cached under
  `~/.cache/termxtract` or `$TERMXTRACT_DATA`.
- The ACTER 1.5 dataset, as `datasets.load("acter")`, in English, French and Dutch, with
  the TermEval 2020 train and test splits, gold terms with their labels, and IOB labels
  (#36).
- The ACL RD-TEC 2.0 dataset, as `datasets.load("acl-rd-tec")`: 299 abstracts of
  computational linguistics papers with terms in seven classes, a choice of annotator,
  and lemmas and part-of-speech tags (#39).
- The GENIA term corpus 3.02, as `datasets.load("genia")`: 2,000 MEDLINE abstracts with
  biomedical terms in 36 classes, outermost or nested (#38).
- `BENCHMARKS.md`, with the results of every method on ACTER, the ACL RD-TEC and GENIA
  next to published results, and `benchmarks/english.py`, which reproduces them (#55).
- `lemma_layer` parameter for every unsupervised method except RAKE, which reads each
  word of a Teanga corpus as its lemma, so inflected forms such as "cell" and "cells"
  are counted, and returned, as one term (#70). `pos_pattern` then matches the lemmas,
  and `datasets.gold_terms` takes a `lemma_layer` too, so gold terms can be compared by
  their lemmas.
- `min_frequency` and `word_pattern` parameters for every unsupervised method except
  RAKE (#67). `min_frequency` drops candidates that occur fewer times in the corpus
  (default 1). `word_pattern` drops candidates with a word that does not match a
  regular expression, or a named pattern of `termxtract.candidates.WORD_PATTERNS`:
  `"alphanumeric"` (the default) for words with a letter or digit, or `"atr4s"` for
  ATR4S's noise filter.
- Several reference corpora, as a list passed as `reference_corpus` (#64). Domain
  Pertinence compares the target corpus with each, as Navigli & Velardi do with one
  corpus per contrasting domain; Weirdness and Relevance read them as one corpus.

- `SupervisedTermExtractor.fit(corpus, labels)` and `predict(corpus)`, to train on one
  corpus and extract terms from another, such as another split of a dataset, another
  domain or another language (#43). `predict` returns `ATEResults` with the terms of
  each document, as `extract` does, and applies `pos_pattern`; `extract` is now `fit`
  followed by `predict` on the same corpus. Roger and token classification have
  matching `fit_strings`, `fit_teanga`, `predict_strings` and `predict_teanga` methods.
- Term offsets for Roger and token classification on Teanga input.
- Roger can learn from a list of terms alone: if no candidate is labelled 0, every
  unlabelled candidate counts as a non-term (#43).

### Changed

- *Breaking:* Residual IDF uses the formula of Church & Gale (1999), as in ATR4S and
  JATE 2.0 (#63): log2(D / df) + log2(1 - e^(-cf / D)), computed once over the corpus,
  so a term has the same score in every document. It used to compute
  log(df / (1 + D (1 - e^(-cf)))), the inverse of the residual with the wrong Poisson
  rate, which favoured common n-grams, and multiplied it by each document's term
  frequency. It is also much faster: it counted each n-gram in each document
  separately, taking hours on 2,000 abstracts, and now takes seconds.
- *Breaking:* C-value, Basic and ComboBasic follow their published definitions, as
  implemented in ATR4S (#62), so their scores change. Frequencies and nesting are
  counted over the whole corpus rather than in each document, so a term has the same
  score in every document. A candidate is nested in another when its words are part of
  the other's words, so "art" is no longer nested in "heart", and only candidates that
  neither start nor end with a stopword count as containing others. C-value ranks
  single words, using log2(|t| + 0.1) (Ventura et al., 2013) instead of log2(|t|), which
  gave them all 0. Basic and ComboBasic use log f(t) instead of log(f(t) + 1), and
  single words no longer gain from being nested in longer candidates. Domain coherence,
  which starts from Basic's top 200 terms, lists each of them once.
- *Breaking:* the topic-modelling method follows the Novel Topic Model as implemented
  in ATR4S (#65), so its scores change. Its LDA model is trained on single words without
  stopwords, instead of on every n-gram with them, and only the `top_words` (new
  parameter, default 200) most probable words of each topic count towards a
  candidate's score, so long candidates full of common words no longer come first. A
  candidate scores log f(t) times the sum of its words' largest topic probabilities,
  where f(t) is its frequency in the corpus, not in each document, so a term has the
  same score in every document.
- *Breaking:* the unsupervised methods except RAKE apply their candidate filters before
  scoring, as ATR4S does (#67): only the occurrences that neither start nor end with a
  stopword, match `word_pattern` and, if given, `pos_pattern`, are counted, and only
  candidates count as containing others in C-value, Basic and ComboBasic. `pos_pattern`
  used to filter the terms after scoring, without changing their scores; a term whose
  occurrences do not all match is now scored by the ones that do. Candidates with a
  word without a letter or digit, such as "95 % CI" on Teanga input, are dropped by
  default; pass `word_pattern=None` to keep them. TF-IDF's term frequencies are still
  relative to every n-gram of the document, so its scores only change for terms with
  occurrences that are filtered out.
- *Breaking:* Domain Pertinence orders the candidates that tie at a score of 1, which
  are most of them with a single reference corpus, so their order is no longer
  arbitrary (#64). By default (`tie_break="ratio"`) their scores are raised by between 0
  and 1 according to the ratio of their frequencies in the target and reference
  corpora, so with one reference corpus candidates are ranked as Weirdness ranks them.
  `tie_break="frequency"` orders them by their frequency in the target corpus, and
  `tie_break=None` keeps the previous scores.

### Removed

- *Breaking:* the AdaBoost supervised method (`method="adaboost"`,
  `termxtract.supervised.adaboost`), with its `threshold`, `max_depth`, `n_estimators`,
  `estimator` and `wordnet_lexicon` parameters, and the dependency on `wn` and Open
  English WordNet (#59). It was a loose, undocumented simplification of Vivaldi,
  Màrquez & Rodríguez (2001), with three features, one of them misnamed, and it was
  the weakest and slowest method of the English benchmark (F1@2000 of 17.7 on ACTER).
  #45 proposes a feature-based classifier in its place. `SupervisedTermExtractor`'s
  `method`, whose default was `"adaboost"`, must now be given.

### Fixed

- Roger trains much faster (#60). Each run samples at most `max_negatives` (new, default
  10,000) of the candidates labelled 0, and keeps all those labelled 1. The area under
  the ROC curve does not depend on the share of positives, so on the English benchmark,
  where every candidate is labelled, this trains in 25 s instead of over two hours,
  with similar results. The runs can be trained in parallel with `n_jobs`, and stopped
  early with `patience`. The fitness is computed in single precision. Each run has its
  own seed from `random_state`, so a given seed gives different scores from before.
- `SupervisedTermExtractor` passed its `n` default of 6 to Roger, whose own default is
  2, multiplying the candidates (#60). `n` now defaults to each method's own default:
  2 for Roger and 6 for token classification.
- Domain coherence, which scored every candidate 0, follows Bordea, Buitelaar &
  Polajnar (2013) (#52). It takes the 200 best candidates by Basic, chooses the 50
  content words that occur in at least a quarter of the documents and are most
  associated with them by normalised PMI within `window_size` words, and scores every
  candidate, not only those 200, by its average normalised PMI with those words. The
  numbers of candidates and words are set by the new `seed_terms` and `context_words`
  parameters. On a Teanga corpus with a part-of-speech layer, only nouns, adjectives and
  verbs can be context words. Each document lists only its own candidates, with their
  offsets, instead of every term of the corpus (#66).
- NMF lists only each document's own candidates, with their offsets for Teanga input,
  instead of every term of the corpus in every document, so its results no longer grow
  with documents times terms (#66). The scores are unchanged.
- Token classification no longer writes to `./results` and `./logs`: it saved a
  checkpoint of the model and its optimizer state every 500 training steps, several
  gigabytes in all, which were never used (#68). The Trainer now works in a temporary
  directory, saves no checkpoints and reports to no logging integrations.

## [0.1.0] - 2026-11-02

The first release on PyPI. This section lists the changes since the code that was
previously only available from GitHub.

### Added

- `termxtract` package exports `TermExtractor`, `SupervisedTermExtractor` and
  `ATEResults`, loaded lazily so `import termxtract` stays fast, and `__version__`.
- `BaseTermExtractor`, the shared base class of all unsupervised extractors, with the
  common n-gram, TF and IDF helpers.
- `results.offsets` for every extractor that scores per-document n-grams (and RAKE) on
  Teanga input.
- `TermExtractor.METHODS` and `SupervisedTermExtractor.METHODS`, which map each method
  name to its extractor class and parameters, replacing the `if`/`elif` chains. An
  unknown `method` now raises a `ValueError` that lists the available methods.
- `stopwords` parameter for every method except RAKE, with a built-in English list
  (NLTK's, based on Snowball's) in `termxtract.stopwords.ENGLISH_STOPWORDS` (#24).
- `pos_pattern` and `pos_layer` parameters for `TermExtractor` and
  `SupervisedTermExtractor`, which keep only the terms whose part-of-speech tags match
  a pattern, such as the built-in `"noun_phrase"` or `"<ADJ|NOUN>*<NOUN>"`, in a Teanga
  corpus (#33). Tags are read from a `upos` or `pos` layer, as written by Teanga's
  CoNLL-U reader and spaCy service. Scores are unchanged.
- `words_layer` parameter for every method except RAKE. On Teanga input, words are read
  from the `words` layer or, if there is none, from a `tokens` layer.
- Optional `transformers` extra (`pip install 'termxtract[transformers]'`) for the
  token-classification method.
- `threshold` and `wordnet_lexicon` parameters for `SupervisedTermExtractor`.
- README, API reference for every extractor, a rewritten TF-IDF tutorial, and tutorials
  for the reference-corpus and supervised methods at <https://teanga.io/TermXtract/>.
- Apache-2.0 license and full package metadata.

### Changed

- *Breaking:* every method returns terms as `{"term": ..., "score": ...}` dictionaries
  ranked from highest to lowest score. TF-IDF used to return plain strings without scores.
- *Breaking:* RAKE's offsets on Teanga input moved from each term dictionary to
  `results.offsets`.
- *Breaking:* torch, transformers, accelerate and sacremoses are no longer installed by
  default; install the `transformers` extra for the token-classification method.
- *Breaking:* Weirdness, Domain Pertinence and Relevance now use their published
  formulas, so their scores change (#29). Domain Pertinence used to give a score of 0 to
  every term missing from the reference corpus, ranking domain terms last; Relevance
  scores are now between 0 and 1, and Weirdness gives finite scores to terms missing from
  the reference corpus. Both input types now use the same formula.
- *Breaking:* candidates that start or end with an English stopword, such as `of`,
  `is a` or `the system`, are no longer returned by default (#24). The scores of the
  other terms are unchanged. Pass `stopwords=None` for the previous behaviour.
- *Breaking:* Roger follows Azé et al. (2005) (#30). It describes each multi-word
  candidate by 12 association measures between its first words and its last word,
  instead of 9 functions of its frequency, so it no longer ranks every equally frequent
  candidate the same. It learns non-linear ranking functions with a (μ+λ) evolution
  strategy, trains only on labelled candidates, and takes the median over `runs`
  (default 21) independent runs instead of over bootstrap samples, so `subsample_ratio`
  is removed. Single words are no longer ranked, and `n` must be at least 2. New
  parameters: `offspring_size`, `runs` and `random_state`; `population_size` now
  defaults to 20. Roger raises a `ValueError` unless at least one candidate is
  labelled `1` and one `0`; a training sample without positive labels used to get an
  undefined (NaN) fitness, which the search treated as the best (#27).
- WordNet features in the AdaBoost extractor use [`wn`](https://github.com/goodmami/wn)
  and Open English WordNet (`oewn:2025+`, downloaded on first use) instead of NLTK.
- Teanga is installed from PyPI instead of GitHub.

### Fixed

- Extractors no longer modify the input Teanga corpus, so several extractors can be run
  on the same `Corpus` object.
- RAKE no longer crashes on Teanga input.
- Weirdness, Domain Pertinence and Relevance raise a clear `TypeError` when the reference
  corpus and the input corpus are of different types, instead of an `AttributeError`
  (#25), and Relevance raises a `ValueError` when no reference corpus is given.
- Token classification only tags the terms labelled `1` as terms, and matches labels
  case-insensitively. Terms labelled `0` used to be tagged as terms too, and labels with
  capital letters were never matched.
- AdaBoost and Roger list candidates with equal scores in order of first occurrence, so
  their results no longer change from run to run.
- NMF, AdaBoost and Roger use the Teanga `words` layer as it is. They used to join the
  n-grams of each document into a string and generate n-grams from it again, which
  created spurious candidates such as `neural neural` and inflated counts, so Teanga
  and plain-text input gave different scores (#32). AdaBoost and Roger lower-case the
  Teanga words, as they do plain text, to match the labels.
- NMF keeps one-letter words such as `a`. It used to drop them, which also created
  n-grams across them, such as `of plant` from `of a plant`.
- Each document's entry in `results.terms` lists only its own terms for the supervised
  methods (#31). AdaBoost and Roger used to list every candidate in the corpus under
  every document; they now list the document's candidates, with their corpus-level
  scores. Token classification lists the terms the model tagged in the document's own
  sentences, instead of every predicted term that occurs as a substring of the
  document, so `cell` is no longer listed for a document that only contains `cells`.
- `alpha=0` and `beta=0` are passed to `basic` and `combobasic` instead of being replaced
  by the default of 0.5. `alpha`, `beta` and `num_topics` no longer accept `None`.
- RIDF no longer crashes when no threshold is given.
- The token-classification extractor works with current transformers releases
  (`eval_strategy`).
- Docstrings no longer raise `SyntaxWarning` on import.

[Unreleased]: https://github.com/TeangaNLP/TermXtract/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/TeangaNLP/TermXtract/releases/tag/v0.1.0
