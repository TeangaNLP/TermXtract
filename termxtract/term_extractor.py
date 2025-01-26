from typing import Union, List, Optional
from teanga import Corpus
from .utils import ATEResults
from .tfidf import TFIDFTermExtractor
from .ridf import RIDFTermExtractor
from .basic import BasicTermExtractor
from .combobasic import ComboBasicTermExtractor
from .cvalue import CValueTermExtractor
from .rake import RAKETermExtractor

class TermExtractor:
    """A wrapper class for selecting the term extraction method."""

    def __init__(self, method: str = "tfidf", **kwargs):
        """
        Initialize the term extractor with the selected method.

        Args:
            method (str): The term extraction method (e.g., 'tfidf', 'ridf', 'basic', 'rake').
            **kwargs: Additional parameters specific to the chosen method.

        Raises:
            ValueError: If the specified method is not supported.
        """
        self.method = method.lower()

        if self.method == "tfidf":
            self.extractor = TFIDFTermExtractor(**kwargs)
        elif self.method == "ridf":
            self.extractor = RIDFTermExtractor(**kwargs)
        elif self.method == "basic":
            self.extractor = BasicTermExtractor(**kwargs)
        elif self.method == "combobasic":
            self.extractor = ComboBasicTermExtractor(**kwargs)
        elif self.method == "cvalue":
            self.extractor = CValueTermExtractor(**kwargs)
        elif self.method == "rake":
            self.extractor = RAKE(**kwargs)
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
            if hasattr(self.extractor, "extract_terms_teanga"):
                return self.extractor.extract_terms_teanga(corpus)
            else:
                raise ValueError(f"The selected method '{self.method}' does not support Teanga Corpus.")

        elif isinstance(corpus, list) and all(isinstance(doc, str) for doc in corpus):
            if hasattr(self.extractor, "extract_terms_strings"):
                return self.extractor.extract_terms_strings(corpus)
            else:
                raise ValueError(f"The selected method '{self.method}' does not support string-based corpora.")

        else:
            raise ValueError("The corpus must be a Teanga Corpus or a list of strings.")

