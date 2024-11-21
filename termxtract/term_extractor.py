from typing import Optional
from teanga import Corpus
from .tfidf import TFIDFTermExtractor

class TermExtractor:
    """A wrapper class for selecting the term extraction method."""

    def __init__(self, method: str = "tfidf", threshold: Optional[float] = None, n: int = 1):
        """Initialize the term extractor with the specified method and parameters.

        Args:
            method (str): The method for term extraction (e.g., 'tfidf').
            threshold (Optional[float]): Minimum score threshold for term inclusion.
            n (int): Maximum length of n-grams to consider.
        
        Raises:
            ValueError: If the specified method is not supported.
        """
        if method == "tfidf":
            self.extractor = TFIDFTermExtractor(threshold=threshold, n=n)
        else:
            raise ValueError(f"Unknown extraction method: {method}")

    def extract(self, corpus: Corpus) -> dict:
        """Extract terms from a Teanga corpus and add them as a layer to each document.

        Args:
            corpus (Corpus): A Teanga corpus with text and words layers.

        Returns:
            dict: A dictionary where keys are document IDs, and values are lists of 
            dictionaries containing terms and their respective offsets.
        """
        return self.extractor.extract_terms(corpus)
