import math
from collections import Counter

from ..stopwords import Stopwords
from .nested_extractor import NestedTermExtractor


class BasicTermExtractor(NestedTermExtractor):
    """
    Basic term extraction (Bordea et al., 2013), which favours candidates that are part
    of many other candidates. Frequencies and nesting are counted over the corpus.
    """

    def __init__(
        self,
        alpha: float = 0.5,
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
        r"""
        Initialize the Basic extractor.

        Args:
            alpha (float): Weight for \( e_t \), the number of candidates containing
                \( t \). ATR4S uses 0.72, from Astrakhantsev (2015), and Bordea et al.
                (2013) use 3.5; the results of the English benchmark are within a point
                of those with the default of 0.5.
            threshold (Optional[float]): Minimum score for term inclusion.
            n (int): Maximum n-gram size (e.g., 1 for unigrams, 2 for bigrams, etc.).
            stopwords (Union[str, Iterable[str], None]): Drop candidates that start or end
                with a stopword: "english", a list of words, or None.
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
        self.alpha = alpha
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

    def score_terms(
        self, frequencies: Counter[str], containing: dict[str, set[str]]
    ) -> dict[str, float]:
        r"""
        Compute the Basic score of each candidate: \( |t| \log f(t) + \alpha e_t \),
        where \( e_t \) is the number of candidates that contain \( t \), or 0 if \( t \)
        is a single word.

        Args:
            frequencies (Counter[str]): The number of occurrences of each candidate in
                the corpus.
            containing (Dict[str, Set[str]]): For each candidate, the other candidates
                it is nested in.

        Returns:
            Dict[str, float]: Basic score for each candidate.
        """
        # As in ATR4S, only candidates of two or more words gain from being nested
        return {
            term: len(term.split()) * math.log(frequency)
            + self.alpha * (len(containing[term]) if " " in term else 0)
            for term, frequency in frequencies.items()
        }
