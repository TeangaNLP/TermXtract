import re
from typing import List
from teanga import Corpus
from ..utils import ATEResults
from .base_extractor import BaseTermExtractor


class TFIDFTermExtractor(BaseTermExtractor):
    """TF-IDF based term extraction with n-gram support and a threshold option."""

    def extract_terms_teanga(self, corpus: Corpus) -> ATEResults:
        """
        Extract terms from a Teanga corpus using TF-IDF.

        Args:
            corpus (Corpus): A Teanga corpus with a `words` span layer.

        Returns:
            ATEResults: Results with ranked terms and scores, and the character offsets of
            each occurrence of each term.
        """
        ngrams_by_doc = {}
        for doc_id in corpus.doc_ids:
            doc = corpus.doc_by_id(doc_id)
            ngrams_by_doc[doc_id] = self.generate_ngrams_teanga(self.teanga_words(doc))

        corpus_ngrams = [[ngram for ngram, _ in ngrams_with_offsets] for ngrams_with_offsets in ngrams_by_doc.values()]
        idf = self.compute_idf(corpus_ngrams)

        terms_by_doc = []
        offsets_by_doc = []
        for doc_id, ngrams_with_offsets in ngrams_by_doc.items():
            tf = self.compute_tf([ngram for ngram, _ in ngrams_with_offsets])
            doc_terms = self.doc_terms(doc_id, {ngram: tf[ngram] * idf[ngram] for ngram in tf})
            terms_by_doc.append(doc_terms)
            offsets_by_doc.append(self.doc_offsets(doc_id, ngrams_with_offsets, doc_terms))

        return ATEResults(corpus=corpus, terms=terms_by_doc, offsets=offsets_by_doc)

    def extract_terms_strings(self, corpus: List[str]) -> ATEResults:
        """
        Extract terms from a list of strings using TF-IDF.

        Args:
            corpus (List[str]): The documents as plain strings.

        Returns:
            ATEResults: Results with ranked terms and scores.
        """
        tokenized_corpus = [re.findall(r'\b\w+\b', doc.lower()) for doc in corpus]
        corpus_ngrams = [self.generate_ngrams_strings(doc) for doc in tokenized_corpus]
        idf = self.compute_idf(corpus_ngrams)

        terms_by_doc = []
        processed_corpus = []
        for idx, doc_text in enumerate(corpus):
            doc_id = f"doc_{idx}"
            processed_corpus.append({"doc_id": doc_id, "text": doc_text})

            tf = self.compute_tf(corpus_ngrams[idx])
            terms_by_doc.append(self.doc_terms(doc_id, {ngram: tf[ngram] * idf[ngram] for ngram in tf}))

        return ATEResults(corpus=processed_corpus, terms=terms_by_doc)
