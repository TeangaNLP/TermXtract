import re
from collections import Counter, defaultdict
from typing import List, Dict, Optional
from teanga import Corpus
from ..utils import ATEResults
from .base_extractor import BaseTermExtractor


class RAKETermExtractor(BaseTermExtractor):
    """RAKE-based term extraction with support for n-grams and keyword linking."""

    def __init__(self, stoplist: List[str], phrase_delimiters: List[str], threshold: Optional[float] = None):
        """
        Initialize the RAKE extractor.

        Args:
            stoplist (List[str]): List of stop words.
            phrase_delimiters (List[str]): List of delimiters to split phrases (e.g., punctuation).
            threshold (Optional[float]): Minimum score for term inclusion.
        """
        super().__init__(threshold=threshold)
        self.stoplist = set(stoplist)
        self.phrase_delimiters = set(phrase_delimiters)

    def generate_candidate_keywords(self, text: str) -> List[List[str]]:
        """
        Generate candidate keywords by splitting on stop words and phrase delimiters.

        Args:
            text (str): Document text.

        Returns:
            List[List[str]]: Candidate keywords as lists of words.
        """
        delimiters = '|'.join(map(re.escape, self.phrase_delimiters))
        words = re.split(f"({delimiters})", text.lower())
        candidates = []
        current_phrase = []

        for word in words:
            word = word.strip()
            if word in self.stoplist or word in self.phrase_delimiters:
                if current_phrase:
                    candidates.append(current_phrase)
                    current_phrase = []
            elif word:
                current_phrase.append(word)
        if current_phrase:
            candidates.append(current_phrase)
        return candidates

    def compute_word_scores(self, candidates: List[List[str]]) -> Dict[str, float]:
        """
        Compute scores for individual words based on degree and frequency.

        Args:
            candidates (List[List[str]]): Candidate keywords as lists of words.

        Returns:
            Dict[str, float]: Word scores.
        """
        word_frequency = Counter()
        word_degree = Counter()
        for phrase in candidates:
            for word in phrase:
                word_frequency[word] += 1
                word_degree[word] += len(phrase) - 1  # Degree = co-occurrences in the phrase

        # Compute degree-to-frequency ratio
        word_scores = {word: word_degree[word] / word_frequency[word] for word in word_frequency}
        return word_scores

    def compute_candidate_scores(self, candidates: List[List[str]], word_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Compute scores for candidate keywords.

        Args:
            candidates (List[List[str]]): Candidate keywords as lists of words.
            word_scores (Dict[str, float]): Scores for individual words.

        Returns:
            Dict[str, float]: Candidate keyword scores.
        """
        candidate_scores = {}
        for phrase in candidates:
            score = sum(word_scores[word] for word in phrase)
            candidate_scores[" ".join(phrase)] = score
        return candidate_scores

    def link_adjoining_keywords(self, text: str, candidate_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Link adjoining keywords with interior stop words.

        Args:
            text (str): Document text.
            candidate_scores (Dict[str, float]): Scores for original candidate keywords.

        Returns:
            Dict[str, float]: Updated keyword scores including linked keywords.
        """
        linked_keywords = defaultdict(float)
        words = text.lower().split()
        for i, word in enumerate(words):
            if word in candidate_scores:
                j = i + 1
                while j < len(words) and words[j] in self.stoplist:
                    j += 1
                if j < len(words) and words[j] in candidate_scores:
                    linked_keyword = f"{word} {' '.join(words[i + 1:j])} {words[j]}"
                    linked_keywords[linked_keyword] += (
                        candidate_scores[word] + candidate_scores[words[j]]
                    )
        candidate_scores.update(linked_keywords)
        return candidate_scores

    def extract_terms_teanga(self, corpus: Corpus) -> ATEResults:
        """
        Extract terms from a Teanga corpus using RAKE.

        Args:
            corpus: A Teanga Corpus object.

        Returns:
            ATEResults: Results with ranked terms and scores, and the character offsets of
            each occurrence of each term.
        """
        corpus.add_layer_meta("terms", layer_type="span", base="text")
        terms_by_doc = []
        offsets_by_doc = []

        for doc_id in corpus.doc_ids:
            doc = corpus.doc_by_id(doc_id)
            text = doc["text"].text
            text = text[0]

            # Step 1: Generate candidate keywords
            candidates = self.generate_candidate_keywords(text)

            # Step 2: Compute word scores
            word_scores = self.compute_word_scores(candidates)

            # Step 3: Compute candidate keyword scores
            candidate_scores = self.compute_candidate_scores(candidates, word_scores)

            # Step 4: Link adjoining keywords and update scores
            linked_scores = self.link_adjoining_keywords(text, candidate_scores)

            # Step 5: Filter terms by threshold and find their offsets
            doc_terms = self.doc_terms(doc_id, linked_scores)
            terms_by_doc.append(doc_terms)

            lowered = text.lower()
            offsets = []
            for term in doc_terms["terms"]:
                start = lowered.find(term["term"])
                while start != -1:
                    end = start + len(term["term"])
                    offsets.append({"term": term["term"], "offset": (start, end)})
                    start = lowered.find(term["term"], end)
            offsets.sort(key=lambda occurrence: occurrence["offset"])
            offsets_by_doc.append({"doc_id": doc_id, "offsets": offsets})

        return ATEResults(corpus=corpus, terms=terms_by_doc, offsets=offsets_by_doc)

    def extract_terms_strings(self, corpus: List[str]) -> ATEResults:
        """
        Extract terms from a plain list of strings using RAKE.

        Args:
            corpus (List[str]): List of documents as strings.

        Returns:
            ATEResults: Results with ranked terms and scores.
        """
        terms_by_doc = []
        processed_corpus = []

        for idx, doc_text in enumerate(corpus):
            doc_id = f"doc_{idx}"
            processed_corpus.append({"doc_id": doc_id, "text": doc_text})

            # Step 1: Generate candidate keywords
            candidates = self.generate_candidate_keywords(doc_text)

            # Step 2: Compute word scores
            word_scores = self.compute_word_scores(candidates)

            # Step 3: Compute candidate keyword scores
            candidate_scores = self.compute_candidate_scores(candidates, word_scores)

            # Step 4: Link adjoining keywords
            linked_scores = self.link_adjoining_keywords(doc_text, candidate_scores)

            # Step 5: Filter terms by threshold
            terms_by_doc.append(self.doc_terms(doc_id, linked_scores))

        return ATEResults(corpus=processed_corpus, terms=terms_by_doc)

