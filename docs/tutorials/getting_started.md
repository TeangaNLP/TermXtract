# Tutorials

These tutorials cover every method of TermXtract, with examples you can run. Start with the TF-IDF tutorial, which also covers the options that all methods share, such as thresholds, multi-word terms, stopwords, Teanga corpora and part-of-speech patterns.

**Unsupervised methods**, which need only the documents:

- [TF-IDF](tfidf_tutorial.md): TF-IDF and Residual IDF, which find what is distinctive about each document, on plain text and on a Teanga corpus.
- [C-Value and related methods](cvalue_tutorial.md): C-value, NC-value, Basic, ComboBasic and domain coherence, which score terms by their frequency and how they nest in other candidates.
- [Reference corpus methods](reference_corpus_tutorial.md): weirdness, domain pertinence and relevance, which compare your documents with general-language text.
- [Unithood methods](unithood_tutorial.md): PMI, the log-likelihood ratio, χ², Dice and the t-score, which measure how strongly the words of a candidate stick together.
- [Keyphrase methods](keyword_tutorial.md): RAKE, YAKE! and TextRank, SingleRank and PositionRank, from keyphrase extraction.
- [Topic model methods](topic_model_tutorial.md): LDA and NMF.
- [Embedding methods](embedding_tutorial.md): EmbedRank and MDERank, which compare the meanings of candidates and documents, and re-ranking any method by related seed terms.

**Supervised methods**, which learn from labelled examples:

- [Overview](supervised_tutorial.md): training labels, training on one corpus and extracting from another, and Teanga corpora.
- [Roger](supervised/roger.md), [Features](supervised/features.md), [Token classification](supervised/token_classification.md) and [In-context learning](supervised/llm.md).
- [Synthetic training data](supervised/augmentation.md) for any supervised method.

To choose a method, see [Which method should I use?](../benchmarks.md)
