import math
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable
from typing import Any

from teanga import Corpus, Document

from ..candidates import candidate_spans, compile_word_pattern
from ..layers import pos_tags, words_with_offsets
from ..pos import compile_pos_pattern
from ..stopwords import Stopwords, load_stopwords
from ..utils import ATEResults, rank_terms


class BaseTermExtractor(ABC):
    """
    Common interface and helpers for the unsupervised term extractors.

    Subclasses implement `extract_terms_teanga` and `extract_terms_strings`, and build
    their results with `doc_terms` and `doc_offsets` so that every extractor returns
    `ATEResults` in the same format.
    """

    def __init__(
        self,
        threshold: float | None = None,
        n: int = 1,
        stopwords: Stopwords = "english",
        words_layer: str | None = None,
        lemma_layer: str | None = None,
        min_frequency: int = 1,
        word_pattern: str | None = "alphanumeric",
        pos_pattern: str | None = None,
        pos_layer: str | None = None,
    ):
        """
        Initialize the extractor.

        The candidates are the n-grams of the corpus that pass the filters set by
        `stopwords`, `word_pattern`, `pos_pattern` and `min_frequency`. Only the
        occurrences that pass them are counted, so they change the scores as well as
        which terms are returned.

        Args:
            threshold (Optional[float]): Minimum score for term inclusion.
            n (int): Maximum n-gram size (e.g., 1 for unigrams, 2 for bigrams, etc.).
            stopwords (Union[str, Iterable[str], None]): Candidates that start or end with
                one of these words are dropped: "english" (the default) for the built-in
                English list, a list of words, or None to keep every candidate.
            words_layer (Optional[str]): For Teanga input, the span layer holding the
                words, or None for `words` or, failing that, `tokens`.
            lemma_layer (Optional[str]): For Teanga input, a sequence layer, based on the
                words layer, holding the lemma of each word. If given, every word is read
                as its lemma, so inflected forms such as "cell" and "cells" are counted as
                one term, which is reported by its lemmas. None (the default) reads the
                words as written.
            min_frequency (int): Drop candidates that occur fewer times than this in the
                corpus. ATR4S uses 2. The default of 1 keeps every candidate.
            word_pattern (Optional[str]): Drop candidates with a word that does not fully
                match this regular expression, or the name of a pattern in
                `termxtract.candidates.WORD_PATTERNS`: "alphanumeric" (the default)
                drops words without a letter or digit, and "atr4s" is ATR4S's noise
                filter. None keeps every word.
            pos_pattern (Optional[str]): For Teanga input, count only the occurrences
                whose part-of-speech tags match this pattern. Tags are written in angle
                brackets, as in `"<ADJ|NOUN>*<NOUN>"`, or use the name of a pattern in
                `termxtract.pos.POS_PATTERNS`, such as `"noun_phrase"`. None (the
                default) allows every tag sequence.
            pos_layer (Optional[str]): The part-of-speech layer used by `pos_pattern`,
                based on the words layer: None (the default) for `upos` or, if the
                corpus has no `upos` layer, `pos`.
        """
        self.threshold = threshold
        self.n = n
        self.stopwords = load_stopwords(stopwords)
        self.words_layer = words_layer
        self.lemma_layer = lemma_layer
        self.min_frequency = min_frequency
        self.word_pattern = (
            None if word_pattern is None else compile_word_pattern(word_pattern)
        )
        self.pos_pattern = (
            None if pos_pattern is None else compile_pos_pattern(pos_pattern)
        )
        self.pos_layer = pos_layer

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
    def extract_terms_strings(self, corpus: list[str]) -> ATEResults:
        """
        Extract terms from a list of strings.

        Args:
            corpus (List[str]): The documents as plain strings.

        Returns:
            ATEResults: Results with ranked terms and scores.
        """

    def teanga_words(self, doc: Document) -> list[tuple[int, int, str]]:
        """
        Get the words of a Teanga document with their offsets.

        Args:
            doc (Document): A Teanga document with a words layer (see `words_layer`).

        Returns:
            List[Tuple[int, int, str]]: A (start, end, text) tuple for each word, where
            text is the word's lemma if `lemma_layer` is set.
        """
        return words_with_offsets(doc, self.words_layer, self.lemma_layer)

    def candidate_ngrams_teanga(
        self, corpus: Corpus
    ) -> dict[str, list[tuple[str, tuple[int, int]]]]:
        """
        Find the occurrences of the candidates in a Teanga corpus.

        Args:
            corpus (Corpus): A Teanga corpus with a words layer (see `words_layer`) and,
                if `pos_pattern` is set, a part-of-speech layer.

        Returns:
            Dict[str, List[Tuple[str, Tuple[int, int]]]]: For each document ID, every
            occurrence of a candidate, with its start and end character offsets.
        """
        ngrams_by_doc = {}
        for doc_id in corpus.doc_ids:
            doc = corpus.doc_by_id(doc_id)
            words = self.teanga_words(doc)
            tags = (
                None
                if self.pos_pattern is None
                else pos_tags(doc, self.pos_layer, self.words_layer)
            )
            texts = [text for _, _, text in words]
            ngrams_by_doc[doc_id] = [
                (" ".join(texts[i:j]), (words[i][0], words[j - 1][1]))
                for i, j in candidate_spans(
                    texts,
                    self.n,
                    self.stopwords,
                    self.word_pattern,
                    tags,
                    self.pos_pattern,
                )
            ]
        frequent = self.frequent_candidates(
            ngram for ngrams in ngrams_by_doc.values() for ngram, _ in ngrams
        )
        return {
            doc_id: [(ngram, offset) for ngram, offset in ngrams if ngram in frequent]
            for doc_id, ngrams in ngrams_by_doc.items()
        }

    def candidate_ngrams_strings(self, docs: list[list[str]]) -> list[list[str]]:
        """
        Find the occurrences of the candidates in documents split into words.

        Args:
            docs (List[List[str]]): The words of each document.

        Returns:
            List[List[str]]: Every occurrence of a candidate in each document.

        Raises:
            ValueError: If `pos_pattern` is set, since plain text has no tags.
        """
        if self.pos_pattern is not None:
            raise ValueError(
                "pos_pattern needs part-of-speech tags, so it only works with a "
                "Teanga corpus with a part-of-speech layer."
            )
        ngrams_by_doc = [
            [
                " ".join(words[i:j])
                for i, j in candidate_spans(
                    words, self.n, self.stopwords, self.word_pattern
                )
            ]
            for words in docs
        ]
        frequent = self.frequent_candidates(
            ngram for ngrams in ngrams_by_doc for ngram in ngrams
        )
        return [
            [ngram for ngram in ngrams if ngram in frequent] for ngrams in ngrams_by_doc
        ]

    def frequent_candidates(self, ngrams: Iterable[str]) -> set[str]:
        """
        Find the candidates that occur at least `min_frequency` times.

        Args:
            ngrams (Iterable[str]): Every occurrence of a candidate in the corpus.

        Returns:
            Set[str]: The candidates that occur often enough.
        """
        counts = Counter(ngrams)
        return {ngram for ngram, count in counts.items() if count >= self.min_frequency}

    def generate_ngrams_teanga(
        self, words_with_offsets: list[tuple[int, int, str]]
    ) -> list[tuple[str, tuple[int, int]]]:
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
                    ngram = " ".join(words[i : i + j])
                    start_offset = words_with_offsets[i][
                        0
                    ]  # Start offset of the first word in the n-gram
                    end_offset = words_with_offsets[i + j - 1][
                        1
                    ]  # End offset of the last word in the n-gram
                    ngrams.append((ngram, (start_offset, end_offset)))
        return ngrams

    def generate_ngrams_strings(self, words: list[str]) -> list[str]:
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
                    ngram = " ".join(words[i : i + j])
                    ngrams.append(ngram)
        return ngrams

    def compute_tf(
        self, ngrams: list[str], total: int | None = None
    ) -> dict[str, float]:
        """
        Compute the relative frequency of each n-gram in a document.

        Args:
            ngrams (List[str]): The n-grams in the document.
            total (Optional[int]): The number to divide the counts by, or None for the
                number of `ngrams`.

        Returns:
            Dict[str, float]: Count of each n-gram divided by the total number of n-grams.
        """
        if not ngrams:
            return {}
        ngram_count = len(ngrams) if total is None else total
        term_frequencies = Counter(ngrams)
        return {ngram: count / ngram_count for ngram, count in term_frequencies.items()}

    def ngram_total(self, num_words: int) -> int:
        """
        Count the n-grams of up to `n` words in a document, candidates or not.

        Args:
            num_words (int): The number of words of the document.

        Returns:
            int: The number of n-grams.
        """
        return sum(max(0, num_words - k + 1) for k in range(1, self.n + 1))

    def compute_idf(self, corpus_ngrams: list[list[str]]) -> dict[str, float]:
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

    def doc_terms(self, doc_id: str, scores: dict[str, float]) -> dict[str, Any]:
        """
        Build one document's entry in `ATEResults.terms`.

        Args:
            doc_id (str): The document ID.
            scores (Dict[str, float]): The score of each candidate term.

        Returns:
            Dict[str, Any]: `{"doc_id": doc_id, "terms": [{"term": ..., "score": ...}, ...]}`, with
            terms below the threshold or starting or ending with a stopword removed, and the rest ranked from highest to lowest score.
        """
        return {
            "doc_id": doc_id,
            "terms": rank_terms(scores, self.threshold, self.stopwords),
        }

    @staticmethod
    def doc_offsets(
        doc_id: str,
        ngrams_with_offsets: Iterable[tuple[str, tuple[int, int]]],
        doc_terms: dict[str, Any],
    ) -> dict[str, Any]:
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
        offsets = [
            {"term": ngram, "offset": offset}
            for ngram, offset in ngrams_with_offsets
            if ngram in kept
        ]
        return {"doc_id": doc_id, "offsets": offsets}
