import math
import re

from teanga import Corpus

from ..utils import ATEResults
from .base_extractor import BaseTermExtractor


class RIDFTermExtractor(BaseTermExtractor):
    """RIDF-based term extraction with n-gram support."""

    def compute_ridf(self, corpus_ngrams: list[list[str]]) -> dict[str, float]:
        """
        Compute Residual Inverse Document Frequency (RIDF) for n-grams.

        Args:
            corpus_ngrams (List[List[str]]): List of n-grams for each document.

        Returns:
            Dict[str, float]: RIDF values for each n-gram.
        """
        num_docs = len(corpus_ngrams)
        all_ngrams = {ngram for doc in corpus_ngrams for ngram in doc}

        # Expected document frequency under a random model
        expected_doc_freq = {ngram: num_docs * (1 - math.exp(-sum(doc.count(ngram) for doc in corpus_ngrams)))
                             for ngram in all_ngrams}

        # Observed document frequency
        observed_doc_freq = {ngram: sum(1 for doc in corpus_ngrams if ngram in doc) for ngram in all_ngrams}

        # RIDF calculation
        ridf = {}
        for ngram in all_ngrams:
            observed = observed_doc_freq[ngram]
            expected = expected_doc_freq[ngram]
            if observed > 0:
                ridf[ngram] = math.log(observed / (1 + expected))
            else:
                ridf[ngram] = 0.0

        return ridf

    def extract_terms_teanga(self, corpus: Corpus) -> ATEResults:
        """
        Extract terms from a Teanga corpus using RIDF.

        Args:
            corpus: A Teanga Corpus object.

        Returns:
            ATEResults: Results with ranked terms and scores, and term offsets.
        """
        ngrams_by_doc = {}
        for doc_id in corpus.doc_ids:
            doc = corpus.doc_by_id(doc_id)
            ngrams_by_doc[doc_id] = self.generate_ngrams_teanga(self.teanga_words(doc))

        corpus_ngrams = [[ngram for ngram, _ in ngrams_with_offsets] for ngrams_with_offsets in ngrams_by_doc.values()]
        ridf_scores = self.compute_ridf(corpus_ngrams)

        terms_by_doc = []
        offsets_by_doc = []
        for doc_id, ngrams_with_offsets in ngrams_by_doc.items():
            tf = self.compute_tf([ngram for ngram, _ in ngrams_with_offsets])
            doc_terms = self.doc_terms(doc_id, {ngram: tf[ngram] * ridf_scores.get(ngram, 0) for ngram in tf})
            terms_by_doc.append(doc_terms)
            offsets_by_doc.append(self.doc_offsets(doc_id, ngrams_with_offsets, doc_terms))

        return ATEResults(corpus=corpus, terms=terms_by_doc, offsets=offsets_by_doc)

    def extract_terms_strings(self, corpus: list[str]) -> ATEResults:
        """
        Extract terms from a plain list of strings using RIDF.

        Args:
            corpus (List[str]): List of documents as strings.

        Returns:
            ATEResults: Results with ranked terms and scores.
        """
        tokenized_corpus = [re.findall(r'\b\w+\b', doc.lower()) for doc in corpus]
        corpus_ngrams = [self.generate_ngrams_strings(doc) for doc in tokenized_corpus]
        ridf_scores = self.compute_ridf(corpus_ngrams)

        terms_by_doc = []
        processed_corpus = []
        for idx, doc_text in enumerate(corpus):
            doc_id = f"doc_{idx}"
            processed_corpus.append({"doc_id": doc_id, "text": doc_text})

            tf = self.compute_tf(corpus_ngrams[idx])
            terms_by_doc.append(self.doc_terms(doc_id, {ngram: tf[ngram] * ridf_scores.get(ngram, 0) for ngram in tf}))

        return ATEResults(corpus=processed_corpus, terms=terms_by_doc)
