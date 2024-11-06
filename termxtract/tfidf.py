import os
from collections import Counter
import math
from typing import List, Dict

class TFIDFTermExtractor:
    """TF-IDF based term extraction."""

    def compute_tf(self, text: str) -> Dict[str, float]:
        """Compute term frequency for a given text.

        Args:
            text (str): The input text from which to compute term frequency.

        Returns:
            dict: A dictionary of term frequencies where the keys are words and the values are their respective frequencies.
        """
        words = text.split()
        word_count = len(words)
        term_frequencies = Counter(words)

        tf = {word: count / word_count for word, count in term_frequencies.items()}
        return tf

    def compute_idf(self, corpus: List[str]) -> Dict[str, float]:
        """Compute inverse document frequency for a given corpus of documents.

        Args:
            corpus (list of str): A list of documents.

        Returns:
            dict: A dictionary of inverse document frequencies where the keys are words and the values are their respective IDF scores.
        """
        num_docs = len(corpus)
        idf = {}
        all_words = set(word for doc in corpus for word in doc.split())

        for word in all_words:
            doc_count = sum(1 for doc in corpus if word in doc.split())
            idf[word] = math.log(num_docs / (1 + doc_count))

        return idf

    def compute_tfidf(self, corpus: List[str]) -> List[Dict[str, float]]:
        """Compute TF-IDF scores for all terms in a corpus of documents.

        Args:
            corpus (list of str): A list of documents.

        Returns:
            list of dict: A list of dictionaries containing TF-IDF scores for each document in the corpus.
        """
        idf = self.compute_idf(corpus)
        tfidf_list = []

        for text in corpus:
            tf = self.compute_tf(text)
            tfidf = {word: tf[word] * idf[word] for word in tf}
            tfidf_list.append(tfidf)

        return tfidf_list

