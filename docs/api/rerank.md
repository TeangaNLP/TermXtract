# Re-ranking

Re-rankers that revise the scores of any unsupervised method, passed to
[TermExtractor](term_extractor.md) as `rerank`. They embed words with a sentence
encoder (see [Embeddings](embeddings.md)), so they need the optional `transformers`
dependencies: `pip install 'termxtract[transformers]'`.

::: termxtract.rerank
