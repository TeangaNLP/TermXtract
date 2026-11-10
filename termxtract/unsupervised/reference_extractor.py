from abc import abstractmethod
from collections import Counter
from typing import TypeAlias

from teanga import Corpus

from ..candidates import split_words
from ..stopwords import Stopwords
from ..utils import ATEResults
from .base_extractor import BaseTermExtractor

ReferenceCorpus: TypeAlias = Corpus | list[str]
"""A reference corpus: a Teanga corpus, or a list of strings."""


class ReferenceCorpusTermExtractor(BaseTermExtractor):
    """
    Base class for extractors that compare the target corpus with a reference corpus.

    Subclasses implement `score_terms`, which scores every n-gram of the target corpus
    from the n-grams of both corpora. Scores are computed over the whole corpus, so a
    term has the same score in every document it occurs in.

    The reference corpus must have the same type as the corpus passed to `extract`: a
    list of strings for plain-text input, or a Teanga `Corpus` with a `words` layer for
    Teanga input. Several reference corpora can be given as a list of them; Domain
    Pertinence compares the target corpus with each, and the other methods read them as
    one corpus.
    """

    def __init__(
        self,
        reference_corpus: ReferenceCorpus | list[ReferenceCorpus] | None,
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

        Args:
            reference_corpus (Union[Corpus, List[str], List[Union[Corpus, List[str]]]]):
                General-language reference corpus, of the same type as the corpus the
                terms are extracted from, or a list of reference corpora.
            threshold (Optional[float]): Minimum score for term inclusion.
            n (int): Maximum n-gram size (e.g., 1 for unigrams, 2 for bigrams, etc.).
            stopwords (Union[str, Iterable[str], None]): Drop candidates that start or end
                with a stopword: a name in `STOPWORD_LISTS`, a list of words, or None.
            words_layer (Optional[str]): For Teanga input, the span layer holding the
                words, or None for `words` or, failing that, `tokens`.
            lemma_layer (Optional[str]): For Teanga input, a sequence layer holding the
                lemma of each word, to count inflected forms such as "cell" and "cells"
                as one term, or None to use the words as written.
            min_frequency (int): Drop candidates that occur fewer times in the corpus.
            word_pattern (Optional[str]): Drop candidates with a word that does not fully
                match this regular expression or named pattern ("alphanumeric", "atr4s"),
                or None to keep every word.
            pos_pattern (Optional[str]): For Teanga input, count only the occurrences
                whose part-of-speech tags match this pattern, or None.
            pos_layer (Optional[str]): The part-of-speech layer for `pos_pattern`, or
                None for `upos` or, failing that, `pos`.
        """
        super().__init__(
            threshold=threshold,
            n=n,
            stopwords=stopwords,
            words_layer=words_layer,
            lemma_layer=lemma_layer,
            min_frequency=min_frequency,
            word_pattern=word_pattern,
            pos_pattern=pos_pattern,
            pos_layer=pos_layer,
        )
        if reference_corpus is None or (
            isinstance(reference_corpus, list) and not reference_corpus
        ):
            raise ValueError(f"{type(self).__name__} requires a reference corpus.")
        self.reference_corpus = reference_corpus
        if isinstance(reference_corpus, Corpus) or all(
            isinstance(doc, str) for doc in reference_corpus
        ):
            self.reference_corpora: list[ReferenceCorpus] = [reference_corpus]
        else:
            self.reference_corpora = list(reference_corpus)
        if not (
            all(isinstance(ref, Corpus) for ref in self.reference_corpora)
            or all(
                isinstance(ref, list) and all(isinstance(doc, str) for doc in ref)
                for ref in self.reference_corpora
            )
        ):
            raise TypeError(
                "The reference corpus must be a Teanga Corpus, a list of strings, or a "
                "list of reference corpora of one of these types."
            )

    @abstractmethod
    def score_terms(
        self, target_ngrams: list[list[str]], ref_ngrams: list[list[str]]
    ) -> dict[str, float]:
        """
        Score the n-grams of the target corpus.

        Args:
            target_ngrams (List[List[str]]): The n-grams of each target document.
            ref_ngrams (List[List[str]]): The n-grams of each reference document.

        Returns:
            Dict[str, float]: The score of every n-gram in the target corpus.
        """

    def score_corpora(
        self, target_ngrams: list[list[str]], ref_ngrams: list[list[list[str]]]
    ) -> dict[str, float]:
        """
        Score the n-grams of the target corpus against one or more reference corpora.

        By default, the reference corpora are read as one corpus and passed to
        `score_terms`.

        Args:
            target_ngrams (List[List[str]]): The n-grams of each target document.
            ref_ngrams (List[List[List[str]]]): The n-grams of each document of each
                reference corpus.

        Returns:
            Dict[str, float]: The score of every n-gram in the target corpus.
        """
        return self.score_terms(
            target_ngrams, [doc for corpus in ref_ngrams for doc in corpus]
        )

    @staticmethod
    def count_ngrams(corpus_ngrams: list[list[str]]) -> tuple[Counter, int]:
        """
        Count the n-grams of a corpus.

        Args:
            corpus_ngrams (List[List[str]]): The n-grams of each document.

        Returns:
            Tuple[Counter, int]: The count of each n-gram, and the total number of n-grams.
        """
        counts = Counter(ngram for doc in corpus_ngrams for ngram in doc)
        return counts, sum(counts.values())

    def extract_terms_strings(self, corpus: list[str]) -> ATEResults:
        """
        Extract terms from a list of strings.

        Args:
            corpus (List[str]): The documents as plain strings. The reference corpus must
                also be a list of strings.

        Returns:
            ATEResults: Results with ranked terms and scores.

        Raises:
            TypeError: If the reference corpus is a Teanga corpus.
        """
        if isinstance(self.reference_corpora[0], Corpus):
            raise TypeError(
                "The reference corpus is a Teanga Corpus, so the corpus to extract "
                "terms from must also be a Teanga Corpus, not a list of strings."
            )
        target_ngrams = self.candidate_ngrams_strings(
            [split_words(doc) for doc in corpus]
        )
        ref_ngrams = [
            [self.generate_ngrams_strings(split_words(doc)) for doc in reference]
            for reference in self.reference_corpora
        ]
        self._check_words(ref_ngrams)
        scores = self.score_corpora(target_ngrams, ref_ngrams)

        terms_by_doc = [
            self.doc_terms(f"doc_{idx}", {ngram: scores[ngram] for ngram in ngrams})
            for idx, ngrams in enumerate(target_ngrams)
        ]
        return ATEResults(corpus=corpus, terms=terms_by_doc)

    def extract_terms_teanga(self, corpus: Corpus) -> ATEResults:
        """
        Extract terms from a Teanga corpus.

        Args:
            corpus (Corpus): A Teanga corpus with a `words` span layer. The reference corpus
                must also be a Teanga corpus with a `words` layer.

        Returns:
            ATEResults: Results with ranked terms and scores, and term offsets.

        Raises:
            TypeError: If the reference corpus is a list of strings.
        """
        # The corpora are all of one type, checked in __init__
        references = [ref for ref in self.reference_corpora if isinstance(ref, Corpus)]
        if not references:
            raise TypeError(
                "The reference corpus is a list of strings, so the corpus to extract "
                "terms from must also be a list of strings, not a Teanga Corpus."
            )
        ngrams_by_doc = self.candidate_ngrams_teanga(corpus)
        target_ngrams = [
            [ngram for ngram, _ in ngrams_with_offsets]
            for ngrams_with_offsets in ngrams_by_doc.values()
        ]
        ref_ngrams = [
            [
                [
                    ngram
                    for ngram, _ in self.generate_ngrams_teanga(
                        self.teanga_words(reference.doc_by_id(doc_id))
                    )
                ]
                for doc_id in reference.doc_ids
            ]
            for reference in references
        ]
        self._check_words(ref_ngrams)
        scores = self.score_corpora(target_ngrams, ref_ngrams)

        terms_by_doc = []
        offsets_by_doc = []
        for doc_id, ngrams_with_offsets in ngrams_by_doc.items():
            doc_terms = self.doc_terms(
                doc_id, {ngram: scores[ngram] for ngram, _ in ngrams_with_offsets}
            )
            terms_by_doc.append(doc_terms)
            offsets_by_doc.append(
                self.doc_offsets(doc_id, ngrams_with_offsets, doc_terms)
            )

        return ATEResults(corpus=corpus, terms=terms_by_doc, offsets=offsets_by_doc)

    @staticmethod
    def _check_words(ref_ngrams: list[list[list[str]]]) -> None:
        """Check that every reference corpus has some words."""
        for corpus in ref_ngrams:
            if not any(corpus):
                raise ValueError("The reference corpus contains no words.")
