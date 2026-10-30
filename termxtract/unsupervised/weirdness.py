import re
from collections import Counter

from teanga import Corpus

from ..utils import ATEResults
from .base_extractor import BaseTermExtractor


class WeirdnessTermExtractor(BaseTermExtractor):
    """Weirdness-based term extraction with n-gram support."""

    def __init__(
        self,
        reference_corpus: Corpus | list[str],
        threshold: float | None = None,
        n: int = 1,
    ):
        """
        Initialize the Weirdness extractor.

        Args:
            reference_corpus: Reference corpus for computing weirdness (Teanga or list of strings).
            threshold (Optional[float]): Minimum score for term inclusion.
            n (int): Maximum n-gram size (e.g., 1 for unigrams, 2 for bigrams, etc.).
        """
        super().__init__(threshold=threshold, n=n)
        self.reference_corpus = reference_corpus

    def compute_term_frequencies(
        self, corpus_ngrams: list[list[str]]
    ) -> dict[str, float]:
        """Compute term frequencies normalized by the total number of words."""
        total_words = sum(len(doc) for doc in corpus_ngrams)
        term_frequencies = Counter(ngram for doc in corpus_ngrams for ngram in doc)
        return {term: freq / total_words for term, freq in term_frequencies.items()}

    def extract_terms_strings(self, corpus: list[str]) -> ATEResults:
        """
        Extract terms from a list of strings using Weirdness.

        Args:
            corpus (List[str]): List of domain-specific documents as strings.

        Returns:
            ATEResults: Results with ranked terms and scores.
        """
        # Tokenize and generate n-grams for target corpus
        tokenized_corpus = [re.findall(r"\b\w+\b", doc.lower()) for doc in corpus]
        target_ngrams = [self.generate_ngrams_strings(doc) for doc in tokenized_corpus]

        # Tokenize and generate n-grams for reference corpus
        ref_tokenized_corpus = [
            re.findall(r"\b\w+\b", doc.lower()) for doc in self.reference_corpus
        ]
        ref_ngrams = [self.generate_ngrams_strings(doc) for doc in ref_tokenized_corpus]

        # Compute normalized term frequencies (NTF) for target and reference corpora
        target_ntf = self.compute_term_frequencies(target_ngrams)
        ref_ntf = self.compute_term_frequencies(ref_ngrams)

        # Compute Weirdness scores
        terms_by_doc = []
        for idx, doc_text in enumerate(corpus):
            doc_id = f"doc_{idx}"
            doc_ngrams = self.generate_ngrams_strings(tokenized_corpus[idx])
            scores = {
                ngram: target_ntf.get(ngram, 0) / max(ref_ntf.get(ngram, 1e-9), 1e-9)
                for ngram in doc_ngrams
            }
            terms_by_doc.append(self.doc_terms(doc_id, scores))

        return ATEResults(corpus=corpus, terms=terms_by_doc)

    def extract_terms_teanga(self, corpus: Corpus) -> ATEResults:
        """
        Extract terms from a Teanga corpus using Weirdness.

        Args:
            corpus: A Teanga Corpus object.

        Returns:
            ATEResults: Results with ranked terms and scores, and term offsets.
        """
        # Ensure reference corpus is valid and has documents
        if not self.reference_corpus or not self.reference_corpus.doc_ids:  # type: ignore[union-attr]  # see #25
            raise ValueError("Reference corpus must be provided and contain documents.")

        # Generate reference n-grams and compute IDF
        ref_ngrams = []
        for doc_id in self.reference_corpus.doc_ids:  # type: ignore[union-attr]  # see #25
            ref_doc = self.reference_corpus.doc_by_id(doc_id)  # type: ignore[union-attr]  # see #25
            ngrams = [
                ngram
                for ngram, _ in self.generate_ngrams_teanga(self.teanga_words(ref_doc))
            ]
            ref_ngrams.append(ngrams)

        ref_idf = self.compute_idf(ref_ngrams)

        # Generate target n-grams
        ngrams_by_doc = {}
        for doc_id in corpus.doc_ids:
            doc = corpus.doc_by_id(doc_id)
            ngrams_by_doc[doc_id] = self.generate_ngrams_teanga(self.teanga_words(doc))

        # Calculate weirdness scores
        terms_by_doc = []
        offsets_by_doc = []
        for doc_id, ngrams_with_offsets in ngrams_by_doc.items():
            tf = self.compute_tf([ngram for ngram, _ in ngrams_with_offsets])
            scores = {
                ngram: tf[ngram]
                / max(ref_idf.get(ngram, 1e-9), 1e-9)  # Avoid division by zero
                for ngram in tf
            }
            doc_terms = self.doc_terms(doc_id, scores)
            terms_by_doc.append(doc_terms)
            offsets_by_doc.append(
                self.doc_offsets(doc_id, ngrams_with_offsets, doc_terms)
            )

        return ATEResults(corpus=corpus, terms=terms_by_doc, offsets=offsets_by_doc)
