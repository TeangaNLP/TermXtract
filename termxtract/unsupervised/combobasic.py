import math
from collections import Counter

from ..stopwords import Stopwords
from .nested_extractor import NestedTermExtractor


class ComboBasicTermExtractor(NestedTermExtractor):
    """
    ComboBasic term extraction (Astrakhantsev, 2015), which extends Basic by also
    favouring candidates that contain many other candidates. Frequencies and nesting are
    counted over the corpus.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        threshold: float | None = None,
        n: int = 1,
        stopwords: Stopwords = "english",
        words_layer: str | None = None,
    ):
        r"""
        Initialize the ComboBasic extractor.

        Args:
            alpha (float): Weight for \( e_t \), the number of candidates containing
                \( t \). Higher values favour intermediate terms.
            beta (float): Weight for \( e'_t \), the number of candidates contained in
                \( t \). Higher values favour more specific terms. ATR4S uses
                `alpha=0.75` and `beta=0.1`; the results of the English benchmark are
                within a point of those with the defaults of 0.5.
            threshold (Optional[float]): Minimum score for term inclusion.
            n (int): Maximum n-gram size (e.g., 1 for unigrams, 2 for bigrams, etc.).
            stopwords (Union[str, Iterable[str], None]): Drop candidates that start or end
                with a stopword: "english", a list of words, or None.
            words_layer (Optional[str]): For Teanga input, the span layer holding the
                words, or None for `words` or, failing that, `tokens`.
        """
        self.alpha = alpha
        self.beta = beta
        super().__init__(
            threshold=threshold, n=n, stopwords=stopwords, words_layer=words_layer
        )

    def score_terms(
        self, frequencies: Counter[str], containing: dict[str, set[str]]
    ) -> dict[str, float]:
        r"""
        Compute the ComboBasic score of each candidate:
        \( |t| \log f(t) + \alpha e_t + \beta e'_t \), where \( e_t \) is the number
        of candidates that contain \( t \) (0 if \( t \) is a single word) and \( e'_t \)
        the number that \( t \) contains.

        Args:
            frequencies (Counter[str]): The number of occurrences of each candidate in
                the corpus.
            containing (Dict[str, Set[str]]): For each candidate, the other candidates
                it is nested in.

        Returns:
            Dict[str, float]: ComboBasic score for each candidate.
        """
        contains: Counter[str] = Counter(
            longer for longers in containing.values() for longer in longers
        )
        # As in ATR4S, only candidates of two or more words gain from being nested
        return {
            term: len(term.split()) * math.log(frequency)
            + self.alpha * (len(containing[term]) if " " in term else 0)
            + self.beta * contains[term]
            for term, frequency in frequencies.items()
        }
