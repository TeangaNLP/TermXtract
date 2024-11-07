# TF-IDF Tutorial

This tutorial guides you through using the TF-IDF term extraction method in TermXtract.

Ensure you have installed TermXtract and its dependencies.

```python
from termxtract.term_extractor import TermExtractor

corpus = [
    "This is a sample document.",
    "This document is another example."
]

extractor = TermExtractor(method="tfidf", threshold=0.1)
results = extractor.extract(corpus)

for doc_result in results:
    print(doc_result)

