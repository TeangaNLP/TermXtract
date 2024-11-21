import re
from collections import Counter
import math
from typing import List, Dict, Optional, Tuple
from teanga import Corpus
from .base_extractor import BaseTermExtractor

class TFIDFTermExtractor(BaseTermExtractor):
    """TF-IDF based term extraction with n-gram support and a threshold option."""

    def __init__(self, threshold: Optional[float] = None, n: int = 1):
        """Initialize the TFIDFTermExtractor with a threshold and n-gram length.

        Args:
            threshold (Optional[float]): Minimum TF-IDF score to include a term.
            n (int): Maximum length of n-grams to consider (e.g., 1 for unigrams, 2 for bigrams).
        """
        self.threshold = threshold
        self.n = n

    def generate_ngrams(self, words_with_offsets: List[Tuple[int, int, str]]) -> List[Tuple[str, Tuple[int, int]]]:
        """Generate n-grams and their offsets for a list of words with offsets.

        Args:
            words_with_offsets (List[Tuple[int, int, str]]): List of words with (start, end) offsets and text.

        Returns:
            List[Tuple[str, Tuple[int, int]]]: List of n-grams with their start and end offsets.
        """
        ngrams = []
        words = [text for _, _, text in words_with_offsets]
        for i in range(len(words)):
            for j in range(1, self.n + 1):
                if i + j <= len(words):
                    ngram = " ".join(words[i:i + j])
                    start_offset = words_with_offsets[i][0]  # Starting offset of the first word in the n-gram
                    end_offset = words_with_offsets[i + j - 1][1]  # Ending offset of the last word in the n-gram
                    ngrams.append((ngram, (start_offset, end_offset)))
        return ngrams

    def compute_tf(self, ngrams: List[str]) -> Dict[str, float]:
        """Compute term frequency for a list of n-grams."""
        ngram_count = len(ngrams)
        term_frequencies = Counter(ngrams)
        return {ngram: count / ngram_count for ngram, count in term_frequencies.items()}

    def compute_idf(self, corpus_ngrams: List[List[str]]) -> Dict[str, float]:
        """Compute inverse document frequency for a corpus of documents as lists of n-grams."""
        num_docs = len(corpus_ngrams)
        idf = {}
        all_ngrams = set(ngram for doc in corpus_ngrams for ngram in doc)
        for ngram in all_ngrams:
            doc_count = sum(1 for doc in corpus_ngrams if ngram in doc)
            idf[ngram] = math.log(num_docs / (1 + doc_count))
        return idf

    def extract_terms(self, corpus: Corpus) -> Dict[str, List[Dict[str, Tuple[int, int]]]]:
        """Compute TF-IDF scores for terms in a Teanga corpus, add 'terms' layer, and return term offsets.

        Args:
            corpus (Corpus): A Teanga corpus with text and words layers.

        Returns:
            Dict[str, List[Dict[str, Tuple[int, int]]]]: A dictionary where keys are document IDs, and values
            are lists of dictionaries containing terms and their respective offsets.
        """
        corpus.add_layer_meta("terms", layer_type="span", base="text")
        # Step 1: Generate and store n-grams once for each document
        ngrams_by_doc = {}
        for doc_id in corpus.doc_ids:
            doc = corpus.doc_by_id(doc_id)
            words_with_offsets = [(start, end, doc.text[start:end]) for start, end in doc.words]
            ngrams_with_offsets = self.generate_ngrams(words_with_offsets)
            ngrams_by_doc[doc_id] = ngrams_with_offsets

        # Prepare corpus-wide list of n-grams for IDF computation
        corpus_ngrams = [[ngram for ngram, _ in ngrams_with_offsets] for ngrams_with_offsets in ngrams_by_doc.values()]
        idf = self.compute_idf(corpus_ngrams)

        # Step 2: Compute TF-IDF for each document and filter by threshold
        terms_offsets_by_doc = {}
        for doc_id, ngrams_with_offsets in ngrams_by_doc.items():
            doc = corpus.doc_by_id(doc_id)
            ngrams = [ngram for ngram, _ in ngrams_with_offsets]
            tf = self.compute_tf(ngrams)
            tfidf = {ngram: tf[ngram] * idf[ngram] for ngram in tf}

            # Apply threshold filtering if set
            if self.threshold is not None:
                tfidf = {ngram: score for ngram, score in tfidf.items() if score >= self.threshold}

            # Step 3: Add "terms" layer based on filtered TF-IDF terms
            # doc.add_layer_meta("terms", layer_type="span", base="text")
            terms_with_offsets = {}
            for ngram, score in tfidf.items():
                # Get the offsets from the precomputed n-grams with offsets
                offsets = [offset for n, offset in ngrams_with_offsets if n == ngram]
                terms_with_offsets[ngram] = offsets

            # Set the term offsets for the "terms" layer in the document
            doc.terms = [offset for offsets in terms_with_offsets.values() for offset in offsets]

            # Store offsets in dictionary for return
            terms_offsets_by_doc[doc_id] = terms_with_offsets

        return terms_offsets_by_doc
