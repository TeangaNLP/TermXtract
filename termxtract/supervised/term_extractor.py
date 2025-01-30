from typing import Union, List, Dict, Optional
from teanga import Corpus
from ..utils import ATEResults
from .adaboost import AdaBoostTermExtractor
from .roger import RogerTermExtractor


class SupervisedTermExtractor:
    """A wrapper class for selecting the supervised term extraction method."""

    def __init__(
        self,
        method: str = "adaboost",
        population_size: int = 50,
        generations: int = 100,
        subsample_ratio: float = 0.8,
        threshold: Optional[float] = None,
        n: int = 1,
    ):
        """
        Initialize the TermExtractor with the desired supervised method and parameters.

        Args:
            method (str): The term extraction method ('adaboost', 'roger').
            population_size (int): Number of ranking functions (only for 'roger').
            generations (int): Number of iterations for optimization (only for 'roger').
            subsample_ratio (float): Data fraction used in each iteration (only for 'roger').
            threshold (Optional[float]): Minimum score threshold for term inclusion.
            n (int): Maximum length of n-grams to consider.
        """
        if method == "adaboost":
            self.extractor = AdaBoostTermExtractor(threshold=threshold, n=n)
        elif method == "roger":
            self.extractor = RogerTermExtractor(
                population_size=population_size,
                generations=generations,
                subsample_ratio=subsample_ratio,
                n=n,
            )
        else:
            raise ValueError(f"Unknown supervised extraction method: {method}")

    def extract(
        self, corpus: Union[Corpus, List[str]], labels: Dict[str, int]
    ) -> ATEResults:
        """
        Extract terms from a given corpus using supervised learning.

        Args:
            corpus (Union[Corpus, List[str]]): Teanga Corpus object or list of strings.
            labels (Dict[str, int]): Dictionary with terms as keys and binary labels {0, 1}.

        Returns:
            ATEResults: Object containing extracted terms with scores.
        """
        if isinstance(corpus, Corpus):
            return self.extractor.extract_terms_teanga(corpus, labels)
        elif isinstance(corpus, list) and all(isinstance(doc, str) for doc in corpus):
            return self.extractor.extract_terms_strings(corpus, labels)
        else:
            raise ValueError("Corpus must be a Teanga Corpus or a list of strings.")
