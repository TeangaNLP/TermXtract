import re
import math
from collections import Counter
from typing import List, Dict, Optional
from teanga import Corpus
from ..utils import ATEResults
from .base_extractor import BaseTermExtractor


class BasicTermExtractor(BaseTermExtractor):
    """Basic-based term extraction with n-gram support."""

    def __init__(self, alpha: float = 0.5, threshold: Optional[float] = None, n: int = 1):
        """
        Initialize the Basic extractor.

        Args:
            alpha (float): Weight for \( e_t \) (number of terms containing \( t \)).
            threshold (Optional[float]): Minimum score for term inclusion.
            n (int): Maximum n-gram size (e.g., 1 for unigrams, 2 for bigrams, etc.).
        """
        self.alpha = alpha
        super().__init__(threshold=threshold, n=n)

    def compute_basic(self, ngrams: List[str]) -> Dict[str, float]:
        """
        Compute Basic scores for n-grams.

        Args:
            ngrams (List[str]): List of n-grams.

        Returns:
            Dict[str, float]: Basic scores for each n-gram.
        """
        # Frequency of each n-gram
        term_frequencies = Counter(ngrams)

        # Nested term statistics
        term_data = {
            term: {
                "frequency": freq,
                "contained_in": 0,
            }
            for term, freq in term_frequencies.items()
        }

        # Count containment relationships
        for term, data in term_data.items():
            for other_term in term_data.keys():
                if term in other_term and term != other_term:
                    data["contained_in"] += 1

        # Calculate Basic scores
        basic_scores = {}
        for term, data in term_data.items():
            length = len(term.split())
            frequency = data["frequency"]
            e_t = data["contained_in"]

            # Basic formula
            basic_scores[term] = length * math.log(frequency + 1) + self.alpha * e_t

        return basic_scores

    def extract_terms_teanga(self, corpus: Corpus) -> ATEResults:
        """
        Extract terms from a Teanga corpus using Basic.

        Args:
            corpus: A Teanga Corpus object.

        Returns:
            ATEResults: Results with ranked terms and scores, and term offsets.
        """
        corpus.add_layer_meta("terms", layer_type="span", base="text")
        terms_by_doc = []
        offsets_by_doc = []
        for doc_id in corpus.doc_ids:
            doc = corpus.doc_by_id(doc_id)
            ngrams_with_offsets = self.generate_ngrams_teanga(self.teanga_words(doc))
            scores = self.compute_basic([ngram for ngram, _ in ngrams_with_offsets])
            doc_terms = self.doc_terms(doc_id, scores)
            terms_by_doc.append(doc_terms)
            offsets_by_doc.append(self.doc_offsets(doc_id, ngrams_with_offsets, doc_terms))

        return ATEResults(corpus=corpus, terms=terms_by_doc, offsets=offsets_by_doc)

    def extract_terms_strings(self, corpus: List[str]) -> ATEResults:
        """
        Extract terms from a plain list of strings using Basic.

        Args:
            corpus (List[str]): List of documents as strings.

        Returns:
            ATEResults: Results with ranked terms and scores.
        """
        terms_by_doc = []
        processed_corpus = []
        for idx, doc_text in enumerate(corpus):
            doc_id = f"doc_{idx}"
            processed_corpus.append({"doc_id": doc_id, "text": doc_text})

            ngrams = self.generate_ngrams_strings(re.findall(r'\b\w+\b', doc_text.lower()))
            terms_by_doc.append(self.doc_terms(doc_id, self.compute_basic(ngrams)))

        return ATEResults(corpus=processed_corpus, terms=terms_by_doc)
