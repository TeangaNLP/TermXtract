from typing import Union, List, Dict, Optional
from teanga import Corpus
from .adaboost import AdaBoostTermExtractor
from ..utils import ATEResults


class SupervisedTermExtractor:
    """A wrapper class for selecting the supervised term extraction method."""

    def __init__(
        self,
        method: str = "adaboost",
        threshold: Optional[float] = None,
        n: int = 1
    ):
        """
        Initialize the SupervisedTermExtractor.

        Args:
            method (str): The supervised term extraction method (e.g., 'adaboost').
            threshold (Optional[float]): Minimum confidence threshold for term inclusion.
            n (int): Maximum length of n-grams to consider.
        """
        if method == "adaboost":
            self.extractor = AdaBoostTermExtractor(threshold=threshold, n=n)
        else:
            raise ValueError(f"Unknown extraction method: {method}")

    def extract(
        self,
        corpus: Union[Corpus, List[str]],
        labels: Dict[str, int]
    ) -> ATEResults:
        """
        Extract terms from a given corpus using supervised learning.

        Args:
            corpus (Union[Corpus, List[str]]): Either a Teanga Corpus object or a list of strings.
            labels (Dict[str, int]): A dictionary mapping terms to labels (1 for term, 0 for non-term).

        Returns:
            ATEResults: An object containing the corpus, terms, and offsets.
        """
        if isinstance(corpus, Corpus):
            return self.extractor.extract_terms_teanga(corpus, labels)
        elif isinstance(corpus, list) and all(isinstance(doc, str) for doc in corpus):
            return self.extractor.extract_terms_strings(corpus, labels)
        else:
            raise ValueError("The corpus must be a Teanga Corpus or a list of strings.")

