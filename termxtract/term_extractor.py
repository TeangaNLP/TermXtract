from typing import Union, List
from teanga import Corpus
from utils import ATEResults
from .tfidf import TFIDFTermExtractor

class TermExtractor:
    """A wrapper class for selecting the term extraction method."""

    def __init__(self, method: str = "tfidf", threshold: Optional[float] = None, n: int = 1):
        if method == "tfidf":
            self.extractor = TFIDFTermExtractor(threshold=threshold, n=n)
        else:
            raise ValueError(f"Unknown extraction method: {method}")

    def extract(self, corpus: Union[Corpus, List[str]]) -> ATEResults:
        """
        Extract terms from a given corpus.

        Args:
            corpus (Union[Corpus, List[str]]): Either a Teanga Corpus object or a list of strings.

        Returns:
            ATEResults: An object containing the corpus, terms, and offsets (if applicable).

        Raises:
            ValueError: If the corpus type is not supported.
        """
        if isinstance(corpus, Corpus):
            return self.extractor.extract_terms_teanga(corpus)
        elif isinstance(corpus, list) and all(isinstance(doc, str) for doc in corpus):
            return self.extractor.extract_terms_strings(corpus)
        else:
            raise ValueError("The corpus must be a Teanga Corpus or a list of strings.")
