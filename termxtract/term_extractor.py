from typing import Union, List, Optional
from teanga import Corpus
from .utils import ATEResults
from .tfidf import TFIDFTermExtractor
from .ridf import RIDFTermExtractor
from .cvalue import CValueTermExtractor
from .combobasic import ComboBasicTermExtractor


class TermExtractor:
    """A wrapper class for selecting the term extraction method."""

    def __init__(self, method: str = "tfidf", threshold: Optional[float] = None, n: int = 1, **kwargs):
        """
        Initialize the extractor with the specified method.

        Args:
            method (str): Extraction method, either "tfidf", "ridf", "cvalue", or "combobasic".
            threshold (Optional[float]): Minimum score for term inclusion.
            n (int): Maximum n-gram size.
            kwargs: Additional parameters for ComboBasicTermExtractor.
        """
        if method == "tfidf":
            self.extractor = TFIDFTermExtractor(threshold=threshold, n=n)
        elif method == "ridf":
            self.extractor = RIDFTermExtractor(threshold=threshold, n=n)
        elif method == "cvalue":
            self.extractor = CValueTermExtractor(threshold=threshold, n=n)
        elif method == "combobasic":
            self.extractor = ComboBasicTermExtractor(threshold=threshold, n=n, **kwargs)
        else:
            raise ValueError(f"Unknown extraction method: {method}")

    def extract(self, corpus: Union[Corpus, List[str]]) -> ATEResults:
        """
        Extract terms from a given corpus.

        Args:
            corpus (Union[Corpus, List[str]]): Either a Teanga Corpus object or a list of strings.

        Returns:
            ATEResults: Results containing terms and scores.

        Raises:
            ValueError: If the corpus type is not supported.
        """
        if isinstance(corpus, Corpus):
            return self.extractor.extract_terms_teanga(corpus)
        elif isinstance(corpus, list) and all(isinstance(doc, str) for doc in corpus):
            return self.extractor.extract_terms_strings(corpus)
        else:
            raise ValueError("The corpus must be a Teanga Corpus or a list of strings.")
