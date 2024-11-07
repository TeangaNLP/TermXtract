import re
from collections import Counter
import math
from typing import List, Dict, Optional
from .base_extractor import BaseTermExtractor

class TFIDFTermExtractor(BaseTermExtractor):
    """TF-IDF based term extraction with a threshold option."""

    def __init__(self, threshold: Optional[float] = None):
        """Initialize the TFIDFTermExtractor with an optional threshold.

        Args:
            threshold (Optional[float]): Minimum TF-IDF score to include a term. 
                                         Terms with scores below this will be filtered out.
                                         Defaults to None (no filtering).
        """
        self.threshold = threshold
    
    def clean_text(self, text: str) -> str:
        """Remove punctuation from the text and convert it to lowercase.

        Args:
            text (str): The input text.

        Returns:
            str: The cleaned text without punctuation.
        """
        # Remove punctuation using regex and convert to lowercase
        return re.sub(r'[^\w\s]', '', text).lower()
    
    def compute_tf(self, text: str) -> Dict[str, float]:
        """Compute term frequency for a given text.

        Args:
            text (str): The input text from which to compute term frequency.

        Returns:
            Dict[str, float]: A dictionary of term frequencies where the keys are words 
            and the values are their respective frequencies.
        """
        # Clean the text to remove punctuation
        cleaned_text = self.clean_text(text)
        words = cleaned_text.split()
        word_count = len(words)
        term_frequencies = Counter(words)
        return {word: count / word_count for word, count in term_frequencies.items()}

    def compute_idf(self, corpus: List[str]) -> Dict[str, float]:
        """Compute inverse document frequency for a given corpus of documents.

        Args:
            corpus (List[str]): A list of documents.

        Returns:
            Dict[str, float]: A dictionary of inverse document frequencies where the keys 
            are words and the values are their respective IDF scores.
        """
        num_docs = len(corpus)
        idf = {}
        all_words = set(word for doc in corpus for word in self.clean_text(doc).split())
        for word in all_words:
            doc_count = sum(1 for doc in corpus if word in self.clean_text(doc).split())
            idf[word] = math.log(num_docs / (1 + doc_count))
        return idf

    def extract_terms(self, corpus: List[str]) -> List[Dict[str, float]]:
        """Compute TF-IDF scores for all terms in a corpus of documents, applying a threshold if specified.

        Args:
            corpus (List[str]): A list of documents.

        Returns:
            List[Dict[str, float]]: A list of dictionaries containing TF-IDF scores for 
            each document in the corpus, with scores below the threshold filtered out.
        """
        idf = self.compute_idf(corpus)
        tfidf_list = []
        for text in corpus:
            tf = self.compute_tf(text)
            tfidf = {word: tf[word] * idf[word] for word in tf}

            # Apply threshold filtering if a threshold is set
            if self.threshold is not None:
                tfidf = {word: score for word, score in tfidf.items() if score >= self.threshold}
            
            tfidf_list.append(tfidf)
        return tfidf_list
