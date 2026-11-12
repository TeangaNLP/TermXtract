# BaseTermExtractor

All unsupervised extractors inherit from `BaseTermExtractor`. It provides the n-gram, TF and IDF helpers they share, and builds their results so that every extractor returns [`ATEResults`](../results.md) in the same format. To add a new extractor, subclass it and implement `extract_terms_teanga` and `extract_terms_strings`.

::: termxtract.unsupervised.base_extractor.BaseTermExtractor

## NestedTermExtractor

C-value, Basic and ComboBasic score candidates by their frequency and by how they are nested in each other, over the whole corpus. They inherit from `NestedTermExtractor`, which counts the candidates, finds which contain which, and leaves the formula to `score_terms`.

::: termxtract.unsupervised.nested_extractor.NestedTermExtractor

## ContextTermExtractor

Domain coherence and NC-value score candidates by the content words around their occurrences. They inherit from `ContextTermExtractor`, which finds the content words, counts them within a window of each candidate, and leaves the formula to `corpus_scores`.

::: termxtract.unsupervised.context_extractor.ContextTermExtractor
