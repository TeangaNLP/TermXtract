import re
from collections import Counter
import math
from typing import List, Dict
from teanga import Corpus
from ..utils import ATEResults
from .base_extractor import BaseTermExtractor


class CValueTermExtractor(BaseTermExtractor):
    """C-value based term extraction with n-gram support."""

    def compute_cvalue(self, ngrams: List[str]) -> Dict[str, float]:
        """
        Compute C-value for each n-gram.

        Args:
            ngrams (List[str]): List of n-grams.

        Returns:
            Dict[str, float]: C-value for each n-gram.
        """
        # Frequency of each n-gram
        term_frequencies = Counter(ngrams)

        # Compute nested term statistics
        cvalue = {}
        term_data = {term: {"frequency": freq, "nested_in": [], "nested_freq": 0} for term, freq in term_frequencies.items()}

        for term, data in term_data.items():
            for other_term in term_data.keys():
                if term in other_term and term != other_term:
                    data["nested_in"].append(other_term)
                    data["nested_freq"] += term_data[other_term]["frequency"]

            p_t = len(data["nested_in"]) if data["nested_in"] else 1
            nested_adjustment = data["nested_freq"] / p_t if data["nested_in"] else 0

            # C-value formula
            cvalue[term] = math.log2(len(term.split())) * (data["frequency"] - nested_adjustment)

        return cvalue

    def extract_terms_teanga(self, corpus: Corpus) -> ATEResults:
        """
        Extract terms from a Teanga corpus using C-value.

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
            cvalues = self.compute_cvalue([ngram for ngram, _ in ngrams_with_offsets])
            doc_terms = self.doc_terms(doc_id, cvalues)
            terms_by_doc.append(doc_terms)
            offsets_by_doc.append(self.doc_offsets(doc_id, ngrams_with_offsets, doc_terms))

        return ATEResults(corpus=corpus, terms=terms_by_doc, offsets=offsets_by_doc)

    def extract_terms_strings(self, corpus: List[str]) -> ATEResults:
        """
        Extract terms from a plain list of strings using C-value.

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

            doc_ngrams = self.generate_ngrams_strings(re.findall(r'\b\w+\b', doc_text.lower()))
            terms_by_doc.append(self.doc_terms(doc_id, self.compute_cvalue(doc_ngrams)))

        return ATEResults(corpus=processed_corpus, terms=terms_by_doc)
