import re
import math
from collections import Counter
from typing import List, Dict, Optional, Tuple
from .utils import ATEResults


class RelevanceTermExtractor:
    """Relevance-based term extraction with n-gram support."""

    def __init__(self, reference_corpus, threshold: Optional[float] = None, n: int = 1):
        """
        Initialize the Relevance extractor.

        Args:
            reference_corpus: Reference corpus for computing relevance (Teanga or list of strings).
            threshold (Optional[float]): Minimum score for term inclusion.
            n (int): Maximum n-gram size (e.g., 1 for unigrams, 2 for bigrams, etc.).
        """
        self.reference_corpus = reference_corpus
        self.threshold = threshold
        self.n = n

    def generate_ngrams_teanga(self, words_with_offsets: List[Tuple[int, int, str]]) -> List[Tuple[str, Tuple[int, int]]]:
        """Generate n-grams with offsets for a Teanga corpus."""
        ngrams = []
        for i in range(len(words_with_offsets)):
            for j in range(1, self.n + 1):
                if i + j <= len(words_with_offsets):
                    ngram = " ".join(word for _, _, word in words_with_offsets[i:i + j])
                    start_offset = words_with_offsets[i][0]
                    end_offset = words_with_offsets[i + j - 1][1]
                    ngrams.append((ngram, (start_offset, end_offset)))
        return ngrams

    def generate_ngrams_strings(self, words: List[str]) -> List[str]:
        """Generate n-grams for a plain list of strings."""
        ngrams = []
        for i in range(len(words)):
            for j in range(1, self.n + 1):
                if i + j <= len(words):
                    ngram = " ".join(words[i:i + j])
                    ngrams.append(ngram)
        return ngrams

    def compute_tf(self, ngrams: List[str]) -> Dict[str, float]:
        """Compute Term Frequency (TF) for a list of n-grams."""
        term_frequencies = Counter(ngrams)
        total_count = sum(term_frequencies.values())
        return {term: freq / total_count for term, freq in term_frequencies.items()}

    def compute_idf(self, corpus_ngrams: List[List[str]]) -> Dict[str, float]:
        """Compute Inverse Document Frequency (IDF) for a list of n-grams."""
        num_docs = len(corpus_ngrams)
        idf_scores = {}
        all_ngrams = set(ngram for doc in corpus_ngrams for ngram in doc)
        for ngram in all_ngrams:
            doc_count = sum(1 for doc in corpus_ngrams if ngram in doc)
            idf_scores[ngram] = math.log(1 + (num_docs / (1 + doc_count)))
        return idf_scores

    def extract_terms_teanga(self, corpus) -> ATEResults:
        """Extract terms from a Teanga corpus using Relevance."""
        # Generate reference IDF from reference corpus
        ref_ngrams = [
            [ngram for ngram, _ in self.generate_ngrams_teanga(
                [(start, end, doc.text[start:end]) for start, end in ref_doc.words]
            )] for ref_doc in (self.reference_corpus.doc_by_id(doc_id) for doc_id in self.reference_corpus.doc_ids)
        ]
        ref_idf = self.compute_idf(ref_ngrams)

        ngrams_by_doc = {}
        for doc_id in corpus.doc_ids:
            doc = corpus.doc_by_id(doc_id)
            words_with_offsets = [(start, end, doc.text[start:end]) for start, end in doc.words]
            ngrams_with_offsets = self.generate_ngrams_teanga(words_with_offsets)
            ngrams_by_doc[doc_id] = ngrams_with_offsets

        corpus_ngrams = [[ngram for ngram, _ in ngrams_with_offsets] for ngrams_with_offsets in ngrams_by_doc.values()]
        target_idf = self.compute_idf(corpus_ngrams)

        terms_by_doc = []
        for doc_id, ngrams_with_offsets in ngrams_by_doc.items():
            tf = self.compute_tf([ngram for ngram, _ in ngrams_with_offsets])
            scores = {
                ngram: 1 - (math.log2(2 + tf[ngram] * target_idf.get(ngram, 0) / ref_idf.get(ngram, 1)))
                for ngram in tf
            }

            terms = [{"term": ngram, "score": score} for ngram, score in scores.items()
                     if self.threshold is None or score >= self.threshold]
            terms_by_doc.append({"doc_id": doc_id, "terms": terms})

        return ATEResults(corpus=corpus, terms=terms_by_doc)

    def extract_terms_strings(self, corpus: List[str]) -> ATEResults:
        """Extract terms from a plain list of strings using Relevance."""
        # Generate reference IDF from reference corpus
        ref_ngrams = [self.generate_ngrams_strings(re.findall(r'\b\w+\b', doc.lower())) for doc in self.reference_corpus]
        ref_idf = self.compute_idf(ref_ngrams)

        tokenized_corpus = [re.findall(r'\b\w+\b', doc.lower()) for doc in corpus]
        corpus_ngrams = [self.generate_ngrams_strings(doc) for doc in tokenized_corpus]
        target_idf = self.compute_idf(corpus_ngrams)

        terms_by_doc = []
        for idx, doc_text in enumerate(corpus):
            tf = self.compute_tf(self.generate_ngrams_strings(tokenized_corpus[idx]))
            scores = {
                ngram: 1 - (math.log2(2 + tf[ngram] * target_idf.get(ngram, 0) / ref_idf.get(ngram, 1)))
                for ngram in tf
            }

            terms = [{"term": ngram, "score": score} for ngram, score in scores.items()
                     if self.threshold is None or score >= self.threshold]
            terms_by_doc.append({"doc_id": f"doc_{idx}", "terms": terms})

        return ATEResults(corpus=corpus, terms=terms_by_doc)
