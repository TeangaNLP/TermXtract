import re
import math
from collections import Counter
from typing import List, Dict, Tuple, Optional
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

    def extract_terms_teanga(self, corpus) -> ATEResults:
        """
        Extract terms from a Teanga corpus using DomainCoherence.

        Args:
            corpus: A Teanga Corpus object.

        Returns:
            ATEResults: Results with terms and scores.
        """
        # Step 1: Extract top 200 term candidates using Basic
        basic_extractor = BasicTermExtractor(threshold=None, n=self.n)
        basic_results = basic_extractor.extract_terms_teanga(corpus)
        basic_terms = [
            {"term": term["term"], "score": term["score"]}
            for doc in basic_results.terms
            for term in doc["terms"]
        ]
        top_200_terms = sorted(basic_terms, key=lambda x: x["score"], reverse=True)[:200]

        # Step 2: Filter context words
        context_words = self._extract_context_words(corpus, top_200_terms)

        # Step 3: Compute DomainCoherence weight
        term_scores = self._compute_domain_coherence(corpus, top_200_terms, context_words)

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

    def _extract_context_words(self, corpus, top_terms: List[Dict[str, float]]) -> List[str]:
        """
        Extract context words filtered by frequency and NPMI.

        Args:
            corpus: A Teanga Corpus object.
            top_terms (List[Dict[str, float]]): Top term candidates extracted using Basic.

        Returns:
            List[str]: Top 50 context words ranked by NPMI.
        """
        context_counter = Counter()
        term_set = {term["term"] for term in top_terms}

        for doc in corpus.docs:
            words = re.findall(r'\b\w+\b', doc.text.lower())
            for term in term_set:
                term_positions = [i for i, word in enumerate(words) if word == term]
                for pos in term_positions:
                    context = words[max(0, pos - self.window_size): pos] + words[pos + 1: pos + 1 + self.window_size]
                    context_counter.update(context)

        # Calculate probabilities
        total_terms = sum(context_counter.values())
        p_term = {term: context_counter[term] / total_terms for term in term_set}
        p_word = {word: context_counter[word] / total_terms for word in context_counter}

        # Calculate NPMI for context words
        npmi_scores = {}
        for word in context_counter:
            if word not in term_set:
                npmi_sum = 0
                for term in term_set:
                    p_t_and_w = context_counter[word] / total_terms
                    if p_t_and_w > 0:
                        npmi_sum += math.log(p_t_and_w / (p_term[term] * p_word[word])) / math.log(p_t_and_w)
                npmi_scores[word] = npmi_sum / len(term_set)

        # Filter context words
        quarter_docs = max(1, len(corpus.doc_ids) // 4)
        frequent_words = [word for word, count in context_counter.items() if count >= quarter_docs]
        context_words = [
            word for word in frequent_words if word in npmi_scores and npmi_scores[word] > 0
        ]
        return sorted(context_words, key=lambda w: npmi_scores[w], reverse=True)[:50]

    def _compute_domain_coherence(self, corpus, top_terms: List[Dict[str, float]], context_words: List[str]) -> Dict[str, float]:
        """
        Compute DomainCoherence scores for top term candidates.

        Args:
            corpus: A Teanga Corpus object.
            top_terms (List[Dict[str, float]]): Top term candidates.
            context_words (List[str]): Filtered context words.

        Returns:
            Dict[str, float]: DomainCoherence scores for each term candidate.
        """
        term_scores = {}
        for term in top_terms:
            npmi_sum = 0
            for word in context_words:
                p_t_and_w = 0
                for doc in corpus.docs:
                    if term["term"] in doc.text and word in doc.text:
                        p_t_and_w += 1
                p_t_and_w /= len(corpus.docs)
                p_term = sum(1 for doc in corpus.docs if term["term"] in doc.text) / len(corpus.docs)
                p_word = sum(1 for doc in corpus.docs if word in doc.text) / len(corpus.docs)
                if p_t_and_w > 0:
                    npmi_sum += math.log(p_t_and_w / (p_term * p_word)) / math.log(p_t_and_w)
            term_scores[term["term"]] = npmi_sum / len(context_words)

        return term_scores

