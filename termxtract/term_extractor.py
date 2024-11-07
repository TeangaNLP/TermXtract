from typing import List, Dict, Optional
from .tfidf import TFIDFTermExtractor
# from .rake import RAKEExtractor  # Future extractors can be added here

class TermExtractor:
    """A wrapper class for selecting the term extraction method."""

    def __init__(self, method: str = "tfidf", threshold: Optional[float] = None):
        """Initialize the term extractor with the specified method and threshold.

        Args:
            method (str): The method for term extraction (e.g., 'tfidf').
            threshold (Optional[float]): Minimum score threshold for term inclusion.
        
        Raises:
            ValueError: If the specified method is not supported.
        """
        if method == "tfidf":
            self.extractor = TFIDFTermExtractor(threshold=threshold)
        # elif method == "rake":
        #     self.extractor = RAKEExtractor()  # Uncomment once RAKE is implemented
        else:
            raise ValueError(f"Unknown extraction method: {method}")

    def extract(self, corpus: List[str]) -> List[Dict[str, float]]:
        """Extract terms from the corpus using the chosen method.

        Args:
            corpus (List[str]): A list of documents.

        Returns:
            List[Dict[str, float]]: A list of dictionaries containing extracted terms 
            and their scores for each document.
        """
        return self.extractor.extract_terms(corpus)

