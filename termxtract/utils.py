from typing import List, Dict, Optional

class ATEResults:
    """Class to represent the results of term extraction."""
    def __init__(self, corpus, terms: List[Dict[str, List[str]]], offsets: Optional[List[Dict[str, List[Dict[str, any]]]]] = None):
        """
        Initialize the ATEResults object.

        Args:
            corpus: The corpus, either a Teanga corpus or a list of dictionaries with `doc_id` and `text`.
            terms (List[Dict[str, List[str]]]): A list of dictionaries representing extracted terms (with `doc_id` and `terms`).
            offsets (Optional[List[Dict[str, List[Dict[str, any]]]]]): A list of dictionaries representing term offsets for Teanga corpora.
        """
        self.corpus = corpus
        self.terms = terms
        self.offsets = offsets
