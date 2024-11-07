corpus = ["This is a sample document.", "This document is another sample."]

# Use TF-IDF extractor with a threshold of 0.1
tfidf_extractor = TermExtractor(method="tfidf", threshold=0.1)
tfidf_results = tfidf_extractor.extract(corpus)

