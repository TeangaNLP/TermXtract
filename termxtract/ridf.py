import re
from collections import Counter
import math
from typing import List, Dict, Optional, Tuple
from .utils import ATEResults


class RIDFTermExtractor:
    """RIDF-based term extraction with n-gram support."""

    def __init__(self, threshold: Optional[float] = None, n: int = 1):
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

    def compute_tf(self, ngrams: List[str]) -> Dict[str, float]:
        """
        Compute Term Frequency (TF) for a list of n-grams.

        Args:
            ngrams (List[str]): List of n-grams.

        Returns:
            Dict[str, float]: Term frequency for each n-gram.
        """
        ngram_count = len(ngrams)
        term_frequencies = Counter(ngrams)
        return {ngram: count / ngram_count for ngram, count in term_frequencies.items()}

    def compute_ridf(self, corpus_ngrams: List[List[str]]) -> Dict[str, float]:
        """
        Compute Residual Inverse Document Frequency (RIDF) for n-grams.

        Args:
            corpus_ngrams (List[List[str]]): List of n-grams for each document.

        Returns:
            Dict[str, float]: RIDF values for each n-gram.
        """
        num_docs = len(corpus_ngrams)
        all_ngrams = set(ngram for doc in corpus_ngrams for ngram in doc)

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

    def extract_terms_teanga(self, corpus) -> ATEResults:
        """
        Extract terms from a Teanga corpus using RIDF.

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
        ridf_scores = self.compute_ridf(corpus_ngrams)

        terms_by_doc = []
        for doc_id, ngrams_with_offsets in ngrams_by_doc.items():
            ngrams = [ngram for ngram, _ in ngrams_with_offsets]
            tf = self.compute_tf(ngrams)
            scores = {ngram: tf[ngram] * ridf_scores.get(ngram, 0) for ngram in tf}

            terms = [{"term": ngram, "score": score} for ngram, score in scores.items() if score >= self.threshold]
            terms_by_doc.append({"doc_id": doc_id, "terms": terms})

        return ATEResults(corpus=corpus, terms=terms_by_doc)

    def extract_terms_strings(self, corpus: List[str]) -> ATEResults:
        """
        Extract terms from a plain list of strings using RIDF.

        Args:
            corpus (List[str]): List of documents as strings.

        Returns:
            ATEResults: Results with terms and scores.
        """
        tokenized_corpus = [re.findall(r'\b\w+\b', doc.lower()) for doc in corpus]
        corpus_ngrams = [self.generate_ngrams_strings(doc) for doc in tokenized_corpus]
        ridf_scores = self.compute_ridf(corpus_ngrams)

        terms_by_doc = []
        processed_corpus = []
        for idx, doc_text in enumerate(corpus):
            doc_id = f"doc_{idx}"
            processed_corpus.append({"doc_id": doc_id, "text": doc_text})

            doc_ngrams = self.generate_ngrams_strings(tokenized_corpus[idx])
            tf = self.compute_tf(doc_ngrams)
            scores = {ngram: tf[ngram] * ridf_scores.get(ngram, 0) for ngram in tf}

            terms = [{"term": ngram, "score": score} for ngram, score in scores.items() if score >= self.threshold]
            terms_by_doc.append({"doc_id": doc_id, "terms": terms})

        return ATEResults(corpus=processed_corpus, terms=terms_by_doc)
