# BaseTermExtractor

All unsupervised extractors inherit from `BaseTermExtractor`. It provides the n-gram, TF and IDF helpers they share, and builds their results so that every extractor returns [`ATEResults`](../results.md) in the same format. To add a new extractor, subclass it and implement `extract_terms_teanga` and `extract_terms_strings`.

::: termxtract.unsupervised.base_extractor.BaseTermExtractor
