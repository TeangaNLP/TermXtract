import math
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple
from teanga import Corpus, Document
from ..utils import ATEResults, rank_terms


class BaseTermExtractor(ABC):
    """
    Common interface and helpers for the unsupervised term extractors.

    Subclasses implement `extract_terms_teanga` and `extract_terms_strings`, and build
    their results with `doc_terms` and `doc_offsets` so that every extractor returns
    `ATEResults` in the same format.
    """

    def __init__(self, threshold: Optional[float] = None, n: int = 1):
        """
        Initialize the extractor.

        Args:
            threshold (Optional[float]): Minimum score for term inclusion.
            n (int): Maximum n-gram size (e.g., 1 for unigrams, 2 for bigrams, etc.).
        """
        self.threshold = threshold
        self.n = n

    @abstractmethod
    def extract_terms_teanga(self, corpus: Corpus) -> ATEResults:
        """
        Extract terms from a Teanga corpus.

        Args:
            corpus (Corpus): A Teanga corpus with a `words` span layer.

        Returns:
            ATEResults: Results with ranked terms and scores, and the offsets of each term
            where the extractor supports them.
        """

    @abstractmethod
    def extract_terms_strings(self, corpus: List[str]) -> ATEResults:
        """
        Extract terms from a list of strings.

        Args:
            corpus (List[str]): The documents as plain strings.

        Returns:
            ATEResults: Results with ranked terms and scores.
        """

    @staticmethod
    def teanga_words(doc: Document) -> List[Tuple[int, int, str]]:
        """
        Get the words of a Teanga document with their offsets.

        Args:
            doc (Document): A Teanga document with a `words` span layer.

        Returns:
            List[Tuple[int, int, str]]: A (start, end, text) tuple for each word.
        """
        return [(start, end, doc.text[start:end]) for start, end in doc.words]

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
        Compute the relative frequency of each n-gram in a document.

        Args:
            ngrams (List[str]): All n-grams in the document.

        Returns:
            Dict[str, float]: Count of each n-gram divided by the total number of n-grams.
        """
        if not ngrams:
            return {}
        ngram_count = len(ngrams)
        term_frequencies = Counter(ngrams)
        return {ngram: count / ngram_count for ngram, count in term_frequencies.items()}

    def compute_idf(self, corpus_ngrams: List[List[str]]) -> Dict[str, float]:
        """
        Compute the inverse document frequency of each n-gram as log(N / (1 + df)).

        Args:
            corpus_ngrams (List[List[str]]): The n-grams of each document.

        Returns:
            Dict[str, float]: IDF score for each n-gram, where N is the number of documents
            and df is the number of documents containing the n-gram.
        """
        num_docs = len(corpus_ngrams)
        doc_sets = [set(doc) for doc in corpus_ngrams]
        all_ngrams = set().union(*doc_sets)
        idf = {}
        for ngram in all_ngrams:
            doc_count = sum(1 for doc in doc_sets if ngram in doc)
            idf[ngram] = math.log(num_docs / (1 + doc_count))
        return idf

    def doc_terms(self, doc_id: str, scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Build one document's entry in `ATEResults.terms`.

        Args:
            doc_id (str): The document ID.
            scores (Dict[str, float]): The score of each candidate term.

        Returns:
            Dict[str, Any]: `{"doc_id": doc_id, "terms": [{"term": ..., "score": ...}, ...]}`, with
            terms below the threshold removed and the rest ranked from highest to lowest score.
        """
        return {"doc_id": doc_id, "terms": rank_terms(scores, self.threshold)}

    @staticmethod
    def doc_offsets(doc_id: str, ngrams_with_offsets: Iterable[Tuple[str, Tuple[int, int]]], doc_terms: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build one document's entry in `ATEResults.offsets`.

        Args:
            doc_id (str): The document ID.
            ngrams_with_offsets (Iterable[Tuple[str, Tuple[int, int]]]): Every n-gram occurrence in
                the document with its (start, end) character offsets.
            doc_terms (Dict[str, Any]): The document's entry in `ATEResults.terms`.

        Returns:
            Dict[str, Any]: `{"doc_id": doc_id, "offsets": [{"term": ..., "offset": (start, end)}, ...]}`,
            listing each occurrence of an extracted term in document order.
        """
        kept = {term["term"] for term in doc_terms["terms"]}
        offsets = [{"term": ngram, "offset": offset} for ngram, offset in ngrams_with_offsets if ngram in kept]
        return {"doc_id": doc_id, "offsets": offsets}
