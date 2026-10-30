# Changelog

All notable changes to TermXtract are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Until 1.0.0,
breaking changes may happen in minor releases (0.x.0) and are listed here under
**Changed** or **Removed**, marked *breaking*.

## [Unreleased]

Nothing has been published to PyPI yet. This section lists the changes since the code
that was previously only available from GitHub.

### Added

- `termxtract` package exports `TermExtractor`, `SupervisedTermExtractor` and
  `ATEResults`, loaded lazily so `import termxtract` stays fast, and `__version__`.
- `BaseTermExtractor`, the shared base class of all unsupervised extractors, with the
  common n-gram, TF and IDF helpers.
- `results.offsets` for every extractor that scores per-document n-grams (and RAKE) on
  Teanga input.
- Optional `transformers` extra (`pip install 'termxtract[transformers]'`) for the
  token-classification method.
- `threshold` and `wordnet_lexicon` parameters for `SupervisedTermExtractor`.
- README, API reference for every extractor, and a rewritten TF-IDF tutorial at
  <https://teanga.io/TermXtract/>.
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
- RIDF no longer crashes when no threshold is given.
- The token-classification extractor works with current transformers releases
  (`eval_strategy`).
- Docstrings no longer raise `SyntaxWarning` on import.

[Unreleased]: https://github.com/TeangaNLP/TermXtract/commits/main
