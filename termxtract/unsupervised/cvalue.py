import re
from collections import Counter
import math
from typing import List, Dict, Optional, Tuple
from .utils import ATEResults


class CValueTermExtractor:
    """C-value based term extraction with n-gram support."""

    def __init__(self, threshold: Optional[float] = None, n: int = 1):
        """
        Initialize the C-value extractor.

        Args:
            threshold (Optional[float]): Minimum C-value score for terms to be included.
            n (int): Maximum n-gram size (e.g., 1 for unigrams, 2 for bigrams, etc.).
        """
        self.threshold = threshold
        self.n = n

    def generate_ngrams_teanga(self, words_with_offsets: List[Tuple[int, int, str]]) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Generate n-grams with offsets for a Teanga corpus.

        Args:
            words_with_offsets (List[Tuple[int, int, str]]): List of (start, end, text) tuples for words.

        Returns:
            List[Tuple[str, Tuple[int, int]]]: List of n-grams with their start and end offsets.
        """
        ngrams = []
        words = [text for _, _, text in words_with_offsets]
        for i in range(len(words)):
            for j in range(1, self.n + 1):
                if i + j <= len(words):
                    ngram = " ".join(words[i:i + j])
                    start_offset = words_with_offsets[i][0]  # Start offset of the first word in the n-gram
                    end_offset = words_with_offsets[i + j - 1][1]  # End offset of the last word in the n-gram
                    ngrams.append((ngram, (start_offset, end_offset)))
        return ngrams

    def generate_ngrams_strings(self, words: List[str]) -> List[str]:
        """
        Generate n-grams for a plain list of strings.

        Args:
            words (List[str]): List of words in the document.

        Returns:
            List[str]: List of n-grams.
        """
        ngrams = []
        for i in range(len(words)):
            for j in range(1, self.n + 1):
                if i + j <= len(words):
                    ngram = " ".join(words[i:i + j])
                    ngrams.append(ngram)
        return ngrams

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

    def extract_terms_teanga(self, corpus) -> ATEResults:
        """
        Extract terms from a Teanga corpus using C-value.

        Args:
            corpus: A Teanga Corpus object.

        Returns:
            ATEResults: Results with terms and scores.
        """
        corpus.add_layer_meta("terms", layer_type="span", base="text")
        ngrams_by_doc = {}
        for doc_id in corpus.doc_ids:
            doc = corpus.doc_by_id(doc_id)
            words_with_offsets = [(start, end, doc.text[start:end]) for start, end in doc.words]
            ngrams_with_offsets = self.generate_ngrams_teanga(words_with_offsets)
            ngrams_by_doc[doc_id] = ngrams_with_offsets

        corpus_ngrams = [[ngram for ngram, _ in ngrams_with_offsets] for ngrams_with_offsets in ngrams_by_doc.values()]

        terms_by_doc = []
        for doc_id, ngrams_with_offsets in ngrams_by_doc.items():
            ngrams = [ngram for ngram, _ in ngrams_with_offsets]
            cvalues = self.compute_cvalue(ngrams)

            # Filter terms by threshold
            terms = [{"term": term, "score": score} for term, score in cvalues.items() if self.threshold is None or score >= self.threshold]
            terms_by_doc.append({"doc_id": doc_id, "terms": terms})

        return ATEResults(corpus=corpus, terms=terms_by_doc)

    def extract_terms_strings(self, corpus: List[str]) -> ATEResults:
        """
        Extract terms from a plain list of strings using C-value.

        Args:
            corpus (List[str]): List of documents as strings.

        Returns:
            ATEResults: Results with terms and scores.
        """
        tokenized_corpus = [re.findall(r'\b\w+\b', doc.lower()) for doc in corpus]
        corpus_ngrams = [self.generate_ngrams_strings(doc) for doc in tokenized_corpus]

        terms_by_doc = []
        processed_corpus = []
        for idx, doc_text in enumerate(corpus):
            doc_id = f"doc_{idx}"
            processed_corpus.append({"doc_id": doc_id, "text": doc_text})

            doc_ngrams = self.generate_ngrams_strings(tokenized_corpus[idx])
            cvalues = self.compute_cvalue(doc_ngrams)

            # Filter terms by threshold
            terms = [{"term": term, "score": score} for term, score in cvalues.items() if self.threshold is None or score >= self.threshold]
            terms_by_doc.append({"doc_id": doc_id, "terms": terms})

        return ATEResults(corpus=processed_corpus, terms=terms_by_doc)
