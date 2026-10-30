import math
import re

from teanga import Corpus

from ..utils import ATEResults
from .base_extractor import BaseTermExtractor


class RelevanceTermExtractor(BaseTermExtractor):
    """Relevance-based term extraction with n-gram support."""

    def __init__(self, reference_corpus: Corpus | list[str], threshold: float | None = None, n: int = 1):
        """
        Initialize the Relevance extractor.

        Args:
            reference_corpus: Reference corpus for computing relevance (Teanga or list of strings).
            threshold (Optional[float]): Minimum score for term inclusion.
            n (int): Maximum n-gram size (e.g., 1 for unigrams, 2 for bigrams, etc.).
        """
        super().__init__(threshold=threshold, n=n)
        self.reference_corpus = reference_corpus

    def compute_idf(self, corpus_ngrams: list[list[str]]) -> dict[str, float]:
        """Compute the inverse document frequency of each n-gram as log(1 + N / (1 + df))."""
        num_docs = len(corpus_ngrams)
        idf_scores = {}
        all_ngrams = {ngram for doc in corpus_ngrams for ngram in doc}
        for ngram in all_ngrams:
            doc_count = sum(1 for doc in corpus_ngrams if ngram in doc)
            idf_scores[ngram] = math.log(1 + (num_docs / (1 + doc_count)))
        return idf_scores

    def extract_terms_teanga(self, corpus) -> ATEResults:
        """Extract terms from a Teanga corpus using Relevance."""
        # Ensure reference corpus is valid
        if not self.reference_corpus or not self.reference_corpus.doc_ids:
            raise ValueError("Reference corpus is required and must contain documents.")

        # Generate reference IDF from reference corpus
        ref_ngrams = []
        for doc_id in self.reference_corpus.doc_ids:
            ref_doc = self.reference_corpus.doc_by_id(doc_id)
            if not ref_doc or not hasattr(ref_doc, 'words'):
                continue
            ngrams = [ngram for ngram, _ in self.generate_ngrams_teanga(self.teanga_words(ref_doc))]
            ref_ngrams.append(ngrams)

        ref_idf = self.compute_idf(ref_ngrams)

        # Generate n-grams and compute target IDF for the given corpus
        ngrams_by_doc = {}
        for doc_id in corpus.doc_ids:
            doc = corpus.doc_by_id(doc_id)
            if not doc or not hasattr(doc, 'words'):
                continue
            ngrams_by_doc[doc_id] = self.generate_ngrams_teanga(self.teanga_words(doc))

        corpus_ngrams = [[ngram for ngram, _ in ngrams_with_offsets] for ngrams_with_offsets in ngrams_by_doc.values()]
        target_idf = self.compute_idf(corpus_ngrams)

        # Calculate scores for each term
        terms_by_doc = []
        offsets_by_doc = []
        for doc_id, ngrams_with_offsets in ngrams_by_doc.items():
            tf = self.compute_tf([ngram for ngram, _ in ngrams_with_offsets])
            scores = {
                ngram: 1 - (math.log2(2 + tf[ngram] * target_idf.get(ngram, 0) / max(ref_idf.get(ngram, 1), 1e-9)))
                for ngram in tf
            }
            doc_terms = self.doc_terms(doc_id, scores)
            terms_by_doc.append(doc_terms)
            offsets_by_doc.append(self.doc_offsets(doc_id, ngrams_with_offsets, doc_terms))

        return ATEResults(corpus=corpus, terms=terms_by_doc, offsets=offsets_by_doc)

    def extract_terms_strings(self, corpus: list[str]) -> ATEResults:
        """Extract terms from a plain list of strings using Relevance."""
        # Generate reference IDF from reference corpus
        ref_ngrams = [self.generate_ngrams_strings(re.findall(r'\b\w+\b', doc.lower())) for doc in self.reference_corpus]
        ref_idf = self.compute_idf(ref_ngrams)

        tokenized_corpus = [re.findall(r'\b\w+\b', doc.lower()) for doc in corpus]
        corpus_ngrams = [self.generate_ngrams_strings(doc) for doc in tokenized_corpus]
        target_idf = self.compute_idf(corpus_ngrams)

        terms_by_doc = []
        for idx, doc_text in enumerate(corpus):
            tf = self.compute_tf(self.generate_ngrams_strings(tokenized_corpus[idx]))
            scores = {
                ngram: 1 - (math.log2(2 + tf[ngram] * target_idf.get(ngram, 0) / ref_idf.get(ngram, 1)))
                for ngram in tf
            }
            terms_by_doc.append(self.doc_terms(f"doc_{idx}", scores))

        return ATEResults(corpus=corpus, terms=terms_by_doc)
