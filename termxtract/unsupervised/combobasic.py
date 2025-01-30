import re
import math
from collections import Counter
from typing import List, Dict, Optional, Tuple
from ..utils import ATEResults


class ComboBasicTermExtractor:
    """ComboBasic-based term extraction with n-gram support."""

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, threshold: Optional[float] = None, n: int = 1):
        """
        Initialize the ComboBasic extractor.

        Args:
            alpha (float): Weight for \( e_t \) (number of terms containing \( t \)).
            beta (float): Weight for \( e'_t \) (number of terms contained in \( t \)).
            threshold (Optional[float]): Minimum score for term inclusion.
            n (int): Maximum n-gram size (e.g., 1 for unigrams, 2 for bigrams, etc.).
        """
        self.alpha = alpha
        self.beta = beta
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
                    start_offset = words_with_offsets[i][0]
                    end_offset = words_with_offsets[i + j - 1][1]
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

    def compute_combobasic(self, ngrams: List[str]) -> Dict[str, float]:
        """
        Compute ComboBasic scores for n-grams.

        Args:
            ngrams (List[str]): List of n-grams.

        Returns:
            Dict[str, float]: ComboBasic scores for each n-gram.
        """
        # Frequency of each n-gram
        term_frequencies = Counter(ngrams)

        # Nested term statistics
        term_data = {
            term: {
                "frequency": freq,
                "contained_in": 0,
                "contains": 0,
            }
            for term, freq in term_frequencies.items()
        }

        # Count containment relationships
        for term, data in term_data.items():
            for other_term in term_data.keys():
                if term in other_term and term != other_term:
                    data["contained_in"] += 1
                elif other_term in term and term != other_term:
                    data["contains"] += 1

        # Calculate ComboBasic scores
        combobasic_scores = {}
        for term, data in term_data.items():
            length = len(term.split())
            frequency = data["frequency"]
            e_t = data["contained_in"]
            e_prime_t = data["contains"]

            # ComboBasic formula
            combobasic_scores[term] = length * math.log(frequency + 1) + self.alpha * e_t + self.beta * e_prime_t

        return combobasic_scores

    def extract_terms_teanga(self, corpus) -> ATEResults:
        """
        Extract terms from a Teanga corpus using ComboBasic.

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

        terms_by_doc = []
        for doc_id, ngrams_with_offsets in ngrams_by_doc.items():
            ngrams = [ngram for ngram, _ in ngrams_with_offsets]
            combobasic_scores = self.compute_combobasic(ngrams)

            # Filter terms by threshold
            terms = [{"term": term, "score": score} for term, score in combobasic_scores.items() if self.threshold is None or score >= self.threshold]
            terms_by_doc.append({"doc_id": doc_id, "terms": terms})

        return ATEResults(corpus=corpus, terms=terms_by_doc)

    def extract_terms_strings(self, corpus: List[str]) -> ATEResults:
        """
        Extract terms from a plain list of strings using ComboBasic.

        Args:
            corpus (List[str]): List of documents as strings.

        Returns:
            ATEResults: Results with terms and scores.
        """
        tokenized_corpus = [re.findall(r'\b\w+\b', doc.lower()) for doc in corpus]
        terms_by_doc = []
        processed_corpus = []
        for idx, doc_text in enumerate(corpus):
            doc_id = f"doc_{idx}"
            processed_corpus.append({"doc_id": doc_id, "text": doc_text})

            tokenized_words = re.findall(r'\b\w+\b', doc_text.lower())
            ngrams = self.generate_ngrams_strings(tokenized_words)
            combobasic_scores = self.compute_combobasic(ngrams)

            # Filter terms by threshold
            terms = [{"term": term, "score": score} for term, score in combobasic_scores.items() if self.threshold is None or score >= self.threshold]
            terms_by_doc.append({"doc_id": doc_id, "terms": terms})

        return ATEResults(corpus=processed_corpus, terms=terms_by_doc)
