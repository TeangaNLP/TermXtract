import re
import math
from collections import Counter
from typing import List, Dict, Optional, Tuple
from .utils import ATEResults
from .basic import BasicTermExtractor


class DomainCoherenceTermExtractor:
    """DomainCoherence-based term extraction."""

    def __init__(self, threshold: Optional[float] = None, n: int = 1, window_size: int = 5):
        """
        Initialize the DomainCoherence extractor.

        Args:
            threshold (Optional[float]): Minimum score for term inclusion.
            n (int): Maximum n-gram size for candidate terms.
            window_size (int): Size of the context window for NPMI calculation.
        """
        self.threshold = threshold
        self.n = n
        self.window_size = window_size

    def generate_ngrams_teanga(self, words_with_offsets: List[Tuple[int, int, str]]) -> List[str]:
        """Generate n-grams for a Teanga corpus."""
        ngrams = []
        words = [text for _, _, text in words_with_offsets]
        for i in range(len(words)):
            for j in range(1, self.n + 1):
                if i + j <= len(words):
                    ngram = " ".join(words[i:i + j])
                    ngrams.append(ngram)
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

    def extract_terms_teanga(self, corpus) -> ATEResults:
        """Extract terms from a Teanga corpus using DomainCoherence."""

        # Step 1: Extract top 200 candidates using Basic
        basic_extractor = BasicTermExtractor(threshold=None, n=self.n)
        basic_results = basic_extractor.extract_terms_teanga(corpus)

        basic_terms = [
            {"term": term["term"], "score": term["score"]}
            for doc in basic_results.terms
            for term in doc["terms"]
        ]
        top_200_terms = sorted(basic_terms, key=lambda x: x["score"], reverse=True)[:200]

        # Step 2: Extract context words and compute NPMI
        context_words = self._extract_context_words_teanga(corpus, top_200_terms)

        # Step 3: Compute DomainCoherence scores
        term_scores = self._compute_domain_coherence_teanga(corpus, top_200_terms, context_words)

        # Prepare results
        terms_by_doc = []
        for doc_id in corpus.doc_ids:
            terms = [
                {"term": term, "score": score}
                for term, score in term_scores.items()
                if self.threshold is None or score >= self.threshold
            ]
            terms_by_doc.append({"doc_id": doc_id, "terms": terms})

        return ATEResults(corpus=corpus, terms=terms_by_doc)

    def extract_terms_strings(self, corpus: List[str]) -> ATEResults:
        """Extract terms from a plain list of strings using DomainCoherence."""
        tokenized_corpus = [re.findall(r'\b\w+\b', doc.lower()) for doc in corpus]

        # Step 1: Extract top 200 candidates using Basic
        basic_extractor = BasicTermExtractor(threshold=None, n=self.n)
        basic_results = basic_extractor.extract_terms_strings(corpus)

        basic_terms = [
            {"term": term["term"], "score": term["score"]}
            for doc in basic_results.terms
            for term in doc["terms"]
        ]
        top_200_terms = sorted(basic_terms, key=lambda x: x["score"], reverse=True)[:200]

        # Step 2: Extract context words and compute NPMI
        context_words = self._extract_context_words_strings(tokenized_corpus, top_200_terms)

        # Step 3: Compute DomainCoherence scores
        term_scores = self._compute_domain_coherence_strings(tokenized_corpus, top_200_terms, context_words)

        # Prepare results
        terms_by_doc = []
        for idx, doc_text in enumerate(corpus):
            doc_id = f"doc_{idx}"
            terms = [
                {"term": term, "score": score}
                for term, score in term_scores.items()
                if self.threshold is None or score >= self.threshold
            ]
            terms_by_doc.append({"doc_id": doc_id, "terms": terms})

        return ATEResults(corpus=corpus, terms=terms_by_doc)

    def _extract_context_words_teanga(self, corpus, top_terms: List[Dict[str, float]]) -> List[str]:
        """Extract context words for a Teanga corpus based on NPMI."""
        context_counter = Counter()
        term_set = {term["term"] for term in top_terms}

        for doc_id in corpus.doc_ids:
            doc = corpus.doc_by_id(doc_id)
            words_with_offsets = [(start, end, doc.text[start:end]) for start, end in doc.words]
            words = [word.lower() for _, _, word in words_with_offsets]

            for term in term_set:
                term_positions = [i for i, word in enumerate(words) if word == term]
                for pos in term_positions:
                    context = words[max(0, pos - self.window_size): pos] + words[pos + 1: pos + 1 + self.window_size]
                    context_counter.update(context)

        # Calculate NPMI scores for context words
        return self._calculate_npmi_scores(context_counter, term_set, len(corpus.doc_ids))

    def _extract_context_words_strings(self, tokenized_corpus: List[List[str]], top_terms: List[Dict[str, float]]) -> List[str]:
        """Extract context words for a list of strings based on NPMI."""
        context_counter = Counter()
        term_set = {term["term"] for term in top_terms}

        for words in tokenized_corpus:
            for term in term_set:
                term_positions = [i for i, word in enumerate(words) if word == term]
                for pos in term_positions:
                    context = words[max(0, pos - self.window_size): pos] + words[pos + 1: pos + 1 + self.window_size]
                    context_counter.update(context)

        # Calculate NPMI scores for context words
        return self._calculate_npmi_scores(context_counter, term_set, len(tokenized_corpus))

    def _calculate_npmi_scores(self, context_counter: Counter, term_set: set, num_docs: int) -> List[str]:
        """Calculate NPMI scores for context words and filter top 50."""
        total_terms = sum(context_counter.values())
        p_term = {term: context_counter[term] / total_terms for term in term_set}
        p_word = {word: context_counter[word] / total_terms for word in context_counter}

        npmi_scores = {}
        for word in context_counter:
            if word not in term_set:
                npmi_sum = 0
                for term in term_set:
                    p_t_and_w = context_counter[word] / total_terms
                    if p_t_and_w > 0:
                        npmi_sum += math.log(p_t_and_w / (p_term[term] * p_word[word])) / math.log(p_t_and_w)
                npmi_scores[word] = npmi_sum / len(term_set)

        quarter_docs = max(1, num_docs // 4)
        frequent_words = [word for word, count in context_counter.items() if count >= quarter_docs]
        context_words = [
            word for word in frequent_words if word in npmi_scores and npmi_scores[word] > 0
        ]
        return sorted(context_words, key=lambda w: npmi_scores[w], reverse=True)[:50]

    def _compute_domain_coherence_teanga(self, corpus, top_terms: List[Dict[str, float]], context_words: List[str]) -> Dict[str, float]:
        """Compute DomainCoherence scores for a Teanga corpus."""
        term_scores = {}
        for term_data in top_terms:
            term = term_data["term"]
            term_scores[term] = self._calculate_term_npmi(term, context_words, corpus, is_teanga=True)
        return term_scores

    def _compute_domain_coherence_strings(self, tokenized_corpus: List[List[str]], top_terms: List[Dict[str, float]], context_words: List[str]) -> Dict[str, float]:
        """Compute DomainCoherence scores for a list of strings."""
        term_scores = {}
        for term_data in top_terms:
            term = term_data["term"]
            term_scores[term] = self._calculate_term_npmi(term, context_words, tokenized_corpus, is_teanga=False)
        return term_scores

    def _calculate_term_npmi(self, term: str, context_words: List[str], corpus, is_teanga: bool) -> float:
        """Calculate the NPMI score for a single term."""
        npmi_sum = 0
        for word in context_words:
            p_t_and_w, p_term, p_word = 0, 0, 0
            if is_teanga:
                num_docs = len(corpus.doc_ids)
                for doc_id in corpus.doc_ids:
                    doc = corpus.doc_by_id(doc_id)
                    words = [doc.text[start:end].lower() for start, end in doc.words]
                    if term in words and word in words:
                        p_t_and_w += 1
                p_term = sum(1 for doc_id in corpus.doc_ids if term in [doc.text[start:end].lower() for start, end in corpus.doc_by_id(doc_id).words]) / num_docs
                p_word = sum(1 for doc_id in corpus.doc_ids if word in [doc.text[start:end].lower() for start, end in corpus.doc_by_id(doc_id).words]) / num_docs
            else:
                num_docs = len(corpus)
                for words in corpus:
                    if term in words and word in words:
                        p_t_and_w += 1
                p_term = sum(1 for words in corpus if term in words) / num_docs
                p_word = sum(1 for words in corpus if word in words) / num_docs

            p_t_and_w /= num_docs
            if p_t_and_w > 0:
                npmi_sum += math.log(p_t_and_w / (p_term * p_word)) / math.log(p_t_and_w)
        return npmi_sum / len(context_words) if context_words else 0

