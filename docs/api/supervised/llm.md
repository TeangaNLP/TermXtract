# In-Context Learning

Used by [`SupervisedTermExtractor`](../supervised_term_extractor.md) with `method="llm"`. The model is reached through an [LLM client](../llm.md), and the default "knn" selection of demonstrations needs the optional `transformers` extra for its sentence encoder.

::: termxtract.supervised.llm
