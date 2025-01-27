def extract_terms_teanga(self, corpus) -> ATEResults:
    # Generate reference IDF from reference corpus
    ref_ngrams = []
    for ref_doc_id in self.reference_corpus.doc_ids:
        ref_doc = self.reference_corpus.doc_by_id(ref_doc_id)
        ref_words_with_offsets = [(start, end, ref_doc.text[start:end]) for start, end in ref_doc.words]
        ref_doc_ngrams = [ngram for ngram, _ in self.generate_ngrams_teanga(ref_words_with_offsets)]
        ref_ngrams.append(ref_doc_ngrams)

    ref_idf = self.compute_idf(ref_ngrams)

    # Generate target corpus n-grams and compute target IDF
    ngrams_by_doc = {}
    for doc_id in corpus.doc_ids:
        doc = corpus.doc_by_id(doc_id)
        words_with_offsets = [(start, end, doc.text[start:end]) for start, end in doc.words]
        ngrams_with_offsets = self.generate_ngrams_teanga(words_with_offsets)
        ngrams_by_doc[doc_id] = ngrams_with_offsets

    corpus_ngrams = [[ngram for ngram, _ in ngrams_with_offsets] for ngrams_with_offsets in ngrams_by_doc.values()]
    target_idf = self.compute_idf(corpus_ngrams)

    # Compute terms and scores for each document
    terms_by_doc = []
    for doc_id, ngrams_with_offsets in ngrams_by_doc.items():
        tf = self.compute_tf([ngram for ngram, _ in ngrams_with_offsets])
        scores = {
            ngram: tf[ngram] * ref_idf.get(ngram, 0) / max(target_idf.get(ngram, 1), 1e-9)
            for ngram in tf
        }

        terms = [{"term": ngram, "score": score} for ngram, score in scores.items()
                 if self.threshold is None or score >= self.threshold]
        terms_by_doc.append({"doc_id": doc_id, "terms": terms})

    return ATEResults(corpus=corpus, terms=terms_by_doc)
