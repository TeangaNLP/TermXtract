from abc import ABC, abstractmethod
from typing import List, Dict

class BaseTermExtractor(ABC):
    @abstractmethod
    def extract_terms(self, corpus: List[str]) -> List[Dict[str, float]]:
        """Extract terms from a given corpus of documents.
        
        Args:
            corpus (List[str]): A list of documents.

        Returns:
            List[Dict[str, float]]: A list of dictionaries containing terms and their scores for each document.
        """
        pass

