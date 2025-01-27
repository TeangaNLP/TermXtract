import re
from collections import Counter
from typing import List, Dict, Optional, Tuple
from .utils import ATEResults


class DomainPertinenceTermExtractor:
    """Domain Pertinence-based term extraction."""

    def __init__(self, threshold: Optional[float] = None, n: int = 1):
        """
        Initialize the Domain Pertinence extractor.

        Args:
            threshold (Optional[float]): Minimum Domain Pertinence score for term inclusion.
            n (int): Maximum n-gram size (e.g., 1 for unigrams, 2 for bigrams, etc.).
        """
        self.threshold = threshold
        self.n = n

    def generate_ngrams_teanga(self, words_with_offsets: List[Tuple[int, int, str]]) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Generate n-grams with offsets for a Teanga corpus.

        Args:
            words_with_offsets (List[Tuple[int, int, str]]): List of (start, end, text) tuples for words.

        Returns:
            List[Tuple[str, Tuple[int, int]]]: List of n-grams with their start and end offsets.
        """
        ngrams = []
        words = [text for _, _, text in words_with_offsets]
        for i in range(len(words)):
            for j in range(1, self.n + 1):
                if i + j <= len(words):
                    ngram = " ".join(words[i:i + j])
                    start_offset = words_with_offsets[i][0]
                    end_offset = words_with_offsets[i + j - 1][1]
                    ngrams.append((ngram, (start_offset, end_offset)))
        return ngrams

    def generate_ngrams_strings(self, words: List[str]) -> List[str]:
        """
        Generate n-grams for a plain list of strings.

        Args:
            words (List[str]): List of words in the document.

        Returns:
            List[str]: List of n-grams.
        """
        ngrams = []
        for i in range(len(words)):
            for j in range(1, self.n + 1):
                if i + j <= len(words):
                    ngram = " ".join(words[i:i + j])
                    ngrams.append(ngram)
        return ngrams

    def compute_term_frequencies(self, corpus: List[List[str]]) -> Dict[str, float]:
        """
        Compute term frequencies for a given corpus.

        Args:
            corpus (List[List[str]]): Tokenized corpus as a list of word lists.

        Returns:
            Dict[str, float]: Term frequencies for each term.
        """
        term_frequencies = Counter(
            ngram
            for doc in corpus
            for ngram in self.generate_ngrams_strings(doc)
        )
        total_terms = sum(term_frequencies.values())
        return {term: freq / total_terms for term, freq in term_frequencies.items()}

    def compute_domain_pertinence(self, target_tf: Dict[str, float], reference_tf: Dict[str, float]) -> Dict[str, float]:
        """
        Compute Domain Pertinence for terms.

        Args:
            target_tf (Dict[str, float]): Term frequencies in the target corpus.
            reference_tf (Dict[str, float]): Term frequencies in the reference corpus.

        Returns:
            Dict[str, float]: Domain Pertinence scores for each term.
        """
        domain_pertinence_scores = {}
        for term, target_freq in target_tf.items():
            reference_freq = reference_tf.get(term, 0.0)
            if reference_freq > 0:
                domain_pertinence_scores[term] = target_freq / reference_freq
            else:
                domain_pertinence_scores[term] = 0.0
        return domain_pertinence_scores

    def extract_terms_teanga(self, target_corpus, reference_corpus) -> ATEResults:
        """
        Extract terms from a Teanga corpus using Domain Pertinence.

        Args:
            target_corpus: A Teanga Corpus object (domain-specific corpus).
            reference_corpus: A Teanga Corpus object (reference/general corpus).

        Returns:
            ATEResults: Results with terms and scores.
        """
        target_corpus.add_layer_meta("terms", layer_type="span", base="text")
        reference_corpus.add_layer_meta("terms", layer_type="span", base="text")

        # Compute term frequencies for reference corpus
        reference_ngrams = [
            ngram
            for doc_id in reference_corpus.doc_ids
            for ngram, _ in self.generate_ngrams_teanga([
                (start, end, reference_corpus.doc_by_id(doc_id).text[start:end])
                for start, end in reference_corpus.doc_by_id(doc_id).words
            ])
        ]
        reference_tf = Counter(reference_ngrams)
        total_reference_ngrams = sum(reference_tf.values())
        reference_tf = {term: freq / total_reference_ngrams for term, freq in reference_tf.items()}

        # Compute term frequencies for target corpus
        ngrams_by_doc = {}
        for doc_id in target_corpus.doc_ids:
            doc = target_corpus.doc_by_id(doc_id)
            words_with_offsets = [(start, end, doc.text[start:end]) for start, end in doc.words]
            ngrams_with_offsets = self.generate_ngrams_teanga(words_with_offsets)
            ngrams_by_doc[doc_id] = ngrams_with_offsets

        target_tf = Counter(
            ngram
            for ngrams_with_offsets in ngrams_by_doc.values()
            for ngram, _ in ngrams_with_offsets
        )
        total_target_ngrams = sum(target_tf.values())
        target_tf = {term: freq / total_target_ngrams for term, freq in target_tf.items()}

        # Compute Domain Pertinence scores
        domain_pertinence_scores = self.compute_domain_pertinence(target_tf, reference_tf)

        # Prepare results
        terms_by_doc = []
        for doc_id, ngrams_with_offsets in ngrams_by_doc.items():
            terms = [
                {"term": ngram, "score": domain_pertinence_scores.get(ngram, 0.0)}
                for ngram, _ in ngrams_with_offsets
                if domain_pertinence_scores.get(ngram, 0.0) >= (self.threshold or 0)
            ]
            terms_by_doc.append({"doc_id": doc_id, "terms": terms})

        return ATEResults(corpus=target_corpus, terms=terms_by_doc)

    def extract_terms_strings(self, target_corpus: List[str], reference_corpus: List[str]) -> ATEResults:
        """
        Extract terms from a plain list of strings using Domain Pertinence.

        Args:
            target_corpus (List[str]): List of documents as strings (domain-specific corpus).
            reference_corpus (List[str]): List of documents as strings (reference/general corpus).

        Returns:
            ATEResults: Results with terms and scores.
        """
        target_tokenized_corpus = [re.findall(r'\b\w+\b', doc.lower()) for doc in target_corpus]
        reference_tokenized_corpus = [re.findall(r'\b\w+\b', doc.lower()) for doc in reference_corpus]

        # Compute term frequencies for reference corpus
        reference_tf = self.compute_term_frequencies(reference_tokenized_corpus)

        # Compute term frequencies for target corpus
        target_tf = self.compute_term_frequencies(target_tokenized_corpus)

        # Compute Domain Pertinence scores
        domain_pertinence_scores = self.compute_domain_pertinence(target_tf, reference_tf)

        # Prepare results
        terms_by_doc = []
        processed_corpus = []
        for idx, doc_text in enumerate(target_corpus):
            doc_id = f"doc_{idx}"
            processed_corpus.append({"doc_id": doc_id, "text": doc_text})

            tokenized_doc = re.findall(r'\b\w+\b', doc_text.lower())
            ngrams = self.generate_ngrams_strings(tokenized_doc)
            terms = [
                {"term": ngram, "score": domain_pertinence_scores.get(ngram, 0.0)}
                for ngram in ngrams
                if domain_pertinence_scores.get(ngram, 0.0) >= (self.threshold or 0)
            ]
            terms_by_doc.append({"doc_id": doc_id, "terms": terms})

        return ATEResults(corpus=processed_corpus, terms=terms_by_doc)

