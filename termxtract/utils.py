from typing import Any, Dict, List, Optional, Union
from teanga import Corpus


def rank_terms(scores: Dict[str, float], threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Turn a dictionary of term scores into the term list used in `ATEResults.terms`.

    Args:
        scores (Dict[str, float]): The score of each candidate term.
        threshold (Optional[float]): If given, terms scoring below it are dropped.

    Returns:
        List[Dict[str, Any]]: `{"term": ..., "score": ...}` dictionaries, ranked from highest
        to lowest score. Terms with equal scores keep their original order.
    """
    terms = [
        {"term": term, "score": float(score)}
        for term, score in scores.items()
        if threshold is None or score >= threshold
    ]
    terms.sort(key=lambda term: term["score"], reverse=True)
    return terms


class ATEResults:
    """
    Class to represent the results of term extraction.

    All extractors return results in the same format:

    - `terms` has one entry per document,
      `{"doc_id": ..., "terms": [{"term": ..., "score": ...}, ...]}`, with the terms
      ranked from highest to lowest score.
    - `offsets` is `None` for plain-text input. For Teanga input, extractors that score
      the n-grams of each document give one entry per document,
      `{"doc_id": ..., "offsets": [{"term": ..., "offset": (start, end)}, ...]}`,
      listing every occurrence of each extracted term.
    """
    def __init__(self, corpus: Union[Corpus, List[str], List[Dict[str, str]]], terms: List[Dict[str, Any]], offsets: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize the ATEResults object.

        Args:
            corpus (Union[Corpus, List[str], List[Dict[str, str]]]): The corpus: a Teanga corpus, the input list of strings, or a list of dictionaries with `doc_id` and `text`.
            terms (List[Dict[str, Any]]): The extracted terms and their scores for each document.
            offsets (Optional[List[Dict[str, Any]]]): The character offsets of each extracted term in each document, for Teanga corpora.
        """
        self.corpus = corpus
        self.terms = terms
        self.offsets = offsets
