from typing import Union, List, Optional
from teanga import Corpus
from .utils import ATEResults
from .tfidf import TFIDFTermExtractor
from .ridf import RIDFTermExtractor
from .basic import BasicTermExtractor
from .combobasic import ComboBasicTermExtractor
from .rake import RAKETermExtractor
from .domainpertinence import DomainPertinenceTermExtractor
from .cvalue import CValueTermExtractor
from .domaincoherence import DomainCoherenceTermExtractor
from .weirdness import WeirdnessTermExtractor


class TermExtractor:
    """A wrapper class for selecting the term extraction method."""

    def __init__(
        self,
        method: str = "tfidf",
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        threshold: Optional[float] = None,
        n: int = 1,
        window_size: int = 5,
        stoplist: Optional[List[str]] = None,
        phrase_delimiters: Optional[List[str]] = None,
        reference_corpus: Optional[Union[Corpus, List[str]]] = None,
    ):
        """
        Initialize the TermExtractor with the desired method and parameters.

        Args:
            method (str): The term extraction method (e.g., 'tfidf', 'ridf', 'basic', 'combobasic', 'rake', 'domainpertinence', 'weirdness').
            alpha (Optional[float]): Weight for \( e_t \) (only for certain methods like 'combobasic').
            beta (Optional[float]): Weight for \( e'_t \) (only for certain methods like 'combobasic').
            threshold (Optional[float]): Minimum score threshold for term inclusion.
            n (int): Maximum length of n-grams to consider.
            stoplist (Optional[List[str]]): List of stop words (used by RAKE).
            phrase_delimiters (Optional[List[str]]): List of phrase delimiters (used by RAKE).
            reference_corpus (Optional[Union[Corpus, List[str]]]): Reference corpus for methods like 'domainpertinence' and 'weirdness'.
        """
        if method == "tfidf":
            self.extractor = TFIDFTermExtractor(threshold=threshold, n=n)
        elif method == "ridf":
            self.extractor = RIDFTermExtractor(threshold=threshold, n=n)
        elif method == "cvalue":
            self.extractor = CValueTermExtractor(threshold=threshold, n=n)
        elif method == "basic":
            self.extractor = BasicTermExtractor(alpha=alpha or 0.5, threshold=threshold, n=n)
        elif method == "combobasic":
            self.extractor = ComboBasicTermExtractor(alpha=alpha or 0.5, beta=beta or 0.5, threshold=threshold, n=n)
        elif method == "domaincoherence":
            self.extractor = DomainCoherenceTermExtractor(window_size=window_size or 5, threshold=threshold, n=n)
        elif method == "rake":
            if stoplist is None or phrase_delimiters is None:
                raise ValueError("RAKE requires both a stoplist and phrase delimiters.")
            self.extractor = RAKETermExtractor(stoplist=stoplist, phrase_delimiters=phrase_delimiters, threshold=threshold)
        elif method == "domainpertinence":
            if reference_corpus is None:
                raise ValueError("DomainPertinence requires a reference corpus.")
            self.extractor = DomainPertinenceTermExtractor(reference_corpus=reference_corpus, threshold=threshold, n=n)
        elif method == "weirdness":
            if reference_corpus is None:
                raise ValueError("Weirdness requires a reference corpus.")
            self.extractor = WeirdnessTermExtractor(reference_corpus=reference_corpus, threshold=threshold, n=n)
        else:
            raise ValueError(f"Unknown extraction method: {method}")

    def extract(
        self,
        corpus: Union[Corpus, List[str]],
    ) -> ATEResults:
        """
        Extract terms from a given corpus.

        Args:
            corpus (Union[Corpus, List[str]]): Either a Teanga Corpus object or a list of strings (target corpus).

        Returns:
            ATEResults: An object containing the corpus, terms, and offsets (if applicable).

        Raises:
            ValueError: If the corpus type is not supported or if a required reference corpus is missing.
        """
        if isinstance(corpus, Corpus):
            return self.extractor.extract_terms_teanga(corpus)
        elif isinstance(corpus, list) and all(isinstance(doc, str) for doc in corpus):
            return self.extractor.extract_terms_strings(corpus)
        else:
            raise ValueError("The corpus must be a Teanga Corpus or a list of strings.")
