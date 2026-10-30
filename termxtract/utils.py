from typing import List, Dict, Optional, Union
from teanga import Corpus

class ATEResults:
    """Class to represent the results of term extraction."""
    def __init__(self, corpus: Union[Corpus, List[str], List[Dict[str, str]]], terms: List[Dict[str, List[str]]], offsets: Optional[List[Dict[str, List[Dict[str, any]]]]] = None):
        """
        Initialize the ATEResults object.

        Args:
            corpus (Union[Corpus, List[str], List[Dict[str, str]]]): The corpus: a Teanga corpus, the input list of strings, or a list of dictionaries with `doc_id` and `text`.
            terms (List[Dict[str, List[str]]]): A list of dictionaries representing extracted terms (with `doc_id` and `terms`).
            offsets (Optional[List[Dict[str, List[Dict[str, any]]]]]): A list of dictionaries representing term offsets for Teanga corpora.
        """
        self.corpus = corpus
        self.terms = terms
        self.offsets = offsets
