# TF-IDF Example

This example demonstrates how to use the `TermExtractor` with the TF-IDF method.

```python
from termxtract.term_extractor import TermExtractor

# Define a sample corpus
corpus = [
    "This is a sample document.",
    "This document is another example."
]

# Create an extractor instance with the TF-IDF method and a threshold
extractor = TermExtractor(method="tfidf", threshold=0.1)

# Extract terms
results = extractor.extract(corpus)

# Print the results
for doc_result in results:
    print(doc_result)
