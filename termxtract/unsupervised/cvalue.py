import math
from collections import Counter

from .nested_extractor import NestedTermExtractor


class CValueTermExtractor(NestedTermExtractor):
    r"""
    C-value term extraction (Frantzi et al., 2000), with the extension of Ventura et al.
    (2013) that also ranks single words, as in ATR4S (Astrakhantsev, 2018).

    A candidate \( t \) of \( |t| \) words with frequency \( f(t) \) in the corpus scores
    \( \log_2(|t| + 0.1) \cdot f(t) \) if no other candidate contains it. Otherwise, the
    average frequency of the set \( T_t \) of candidates that contain it is subtracted:
    \( \log_2(|t| + 0.1) \cdot (f(t) - \frac{1}{|T_t|} \sum_{s \in T_t} f(s)) \).
    """

    def score_terms(
        self, frequencies: Counter[str], containing: dict[str, set[str]]
    ) -> dict[str, float]:
        """
        Compute the C-value of each candidate.

        Args:
            frequencies (Counter[str]): The number of occurrences of each candidate in
                the corpus.
            containing (Dict[str, Set[str]]): For each candidate, the other candidates
                it is nested in.

        Returns:
            Dict[str, float]: C-value for each candidate.
        """
        cvalue = {}
        for term, frequency in frequencies.items():
            longer = containing[term]
            nested = sum(frequencies[s] for s in longer) / len(longer) if longer else 0
            cvalue[term] = math.log2(len(term.split()) + 0.1) * (frequency - nested)
        return cvalue
