import re
from collections import Counter
from typing import List, Dict, Optional, Tuple
from .utils import ATEResults


class WeirdnessTermExtractor:
    """Weirdness-based term extraction with n-gram support."""

    def __init__(self, reference_corpus, threshold: Optional[float] = None, n: int = 1):
        """
        Initialize the Weirdness extractor.

        Args:
            reference_corpus: Reference corpus for computing weirdness (Teanga or list of strings).
            threshold (Optional[float]): Minimum score for term inclusion.
            n (int): Maximum n-gram size (e.g., 1 for unigrams, 2 for bigrams, etc.).
        """
        self.reference_corpus = reference_corpus
        self.threshold = threshold
        self.n = n

    def generate_ngrams_strings(self, words: List[str]) -> List[str]:
        """Generate n-grams from a list of words."""
        ngrams = []
        for i in range(len(words)):
            for j in range(1, self.n + 1):
                if i + j <= len(words):
                    ngram = " ".join(words[i:i + j])
                    ngrams.append(ngram)
        return ngrams

    def compute_term_frequencies(self, corpus_ngrams: List[List[str]]) -> Dict[str, float]:
        """Compute term frequencies normalized by the total number of words."""
        total_words = sum(len(doc) for doc in corpus_ngrams)
        term_frequencies = Counter(ngram for doc in corpus_ngrams for ngram in doc)
        return {term: freq / total_words for term, freq in term_frequencies.items()}

    def extract_terms_strings(self, corpus: List[str]) -> ATEResults:
        """
        Extract terms from a list of strings using Weirdness.

        Args:
            corpus (List[str]): List of domain-specific documents as strings.

        Returns:
            ATEResults: Results with terms and scores.
        """
        # Tokenize and generate n-grams for target corpus
        tokenized_corpus = [re.findall(r'\b\w+\b', doc.lower()) for doc in corpus]
        target_ngrams = [self.generate_ngrams_strings(doc) for doc in tokenized_corpus]

        # Tokenize and generate n-grams for reference corpus
        ref_tokenized_corpus = [re.findall(r'\b\w+\b', doc.lower()) for doc in self.reference_corpus]
        ref_ngrams = [self.generate_ngrams_strings(doc) for doc in ref_tokenized_corpus]

        # Compute normalized term frequencies (NTF) for target and reference corpora
        target_ntf = self.compute_term_frequencies(target_ngrams)
        ref_ntf = self.compute_term_frequencies(ref_ngrams)

        # Compute Weirdness scores
        terms_by_doc = []
        for idx, doc_text in enumerate(corpus):
            doc_id = f"doc_{idx}"
            doc_ngrams = self.generate_ngrams_strings(tokenized_corpus[idx])
            scores = {
                ngram: target_ntf.get(ngram, 0) / max(ref_ntf.get(ngram, 1e-9), 1e-9)
                for ngram in doc_ngrams
            }

            # Filter terms by threshold
            terms = [{"term": ngram, "score": score} for ngram, score in scores.items()
                     if self.threshold is None or score >= self.threshold]
            terms_by_doc.append({"doc_id": doc_id, "terms": terms})

        return ATEResults(corpus=corpus, terms=terms_by_doc)
    
    def extract_terms_teanga(self, corpus) -> ATEResults:
        """
        Extract terms from a Teanga corpus using Weirdness.
    
        Args:
            corpus: A Teanga Corpus object.
    
        Returns:
            ATEResults: Results with terms and scores.
        """
        # Ensure reference corpus is valid and has documents
        if not self.reference_corpus or not self.reference_corpus.doc_ids:
            raise ValueError("Reference corpus must be provided and contain documents.")
    
        # Generate reference n-grams and compute IDF
        ref_ngrams = []
        for doc_id in self.reference_corpus.doc_ids:
            ref_doc = self.reference_corpus.doc_by_id(doc_id)
            words_with_offsets = [(start, end, ref_doc.text[start:end]) for start, end in ref_doc.words]
            ngrams = [ngram for ngram, _ in self.generate_ngrams_teanga(words_with_offsets)]
            ref_ngrams.append(ngrams)
    
        ref_idf = self.compute_idf(ref_ngrams)
    
        # Generate target n-grams
        ngrams_by_doc = {}
        for doc_id in corpus.doc_ids:
            doc = corpus.doc_by_id(doc_id)
            words_with_offsets = [(start, end, doc.text[start:end]) for start, end in doc.words]
            ngrams_with_offsets = self.generate_ngrams_teanga(words_with_offsets)
            ngrams_by_doc[doc_id] = ngrams_with_offsets
    
        corpus_ngrams = [[ngram for ngram, _ in ngrams_with_offsets] for ngrams_with_offsets in ngrams_by_doc.values()]
        target_idf = self.compute_idf(corpus_ngrams)
    
        # Calculate weirdness scores
        terms_by_doc = []
        for doc_id, ngrams_with_offsets in ngrams_by_doc.items():
            tf = self.compute_tf([ngram for ngram, _ in ngrams_with_offsets])
            scores = {
                ngram: tf[ngram] / max(ref_idf.get(ngram, 1e-9), 1e-9)  # Avoid division by zero
                for ngram in tf
            }
    
            terms = [{"term": ngram, "score": score} for ngram, score in scores.items()
                     if self.threshold is None or score >= self.threshold]
            terms_by_doc.append({"doc_id": doc_id, "terms": terms})
    
        return ATEResults(corpus=corpus, terms=terms_by_doc)
