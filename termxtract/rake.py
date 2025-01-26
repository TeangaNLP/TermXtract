import re
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple
from .utils import ATEResults


class RAKETermExtractor:
    """RAKE (Rapid Automatic Keyword Extraction) term extractor."""

    def __init__(self, stoplist: List[str], threshold: Optional[float] = None, n: int = 1):
        """
        Initialize the RAKE extractor.

        Args:
            stoplist (List[str]): A list of stopwords to exclude from candidate phrases.
            threshold (Optional[float]): Minimum score for term inclusion.
            n (int): Maximum n-gram size for candidate terms.
        """
        self.stoplist = set(stoplist)
        self.threshold = threshold
        self.n = n

    def _extract_candidates(self, text: str) -> List[List[str]]:
        """
        Extract candidate keywords from the text following RAKE methodology.

        Args:
            text (str): Input document text.

        Returns:
            List[List[str]]: List of candidate keywords as lists of contiguous words.
        """
        # Split the text into words using word delimiters (e.g., spaces, punctuation)
        words = re.findall(r'\b\w+\b', text.lower())

        # Initialize variables for candidate extraction
        candidates = []
        current_candidate = []

        # Iterate through words and split on stopwords or phrase delimiters
        for word in words:
            if word in self.stoplist:
                # Stopword or phrase delimiter: end the current candidate
                if current_candidate:
                    candidates.append(current_candidate)
                    current_candidate = []
            else:
                # Add word to the current candidate
                current_candidate.append(word)

        # Add the last candidate if there is one
        if current_candidate:
            candidates.append(current_candidate)

        return candidates

    def _score_candidates(self, candidates: List[List[str]]) -> Dict[str, float]:
        """
        Compute RAKE scores for candidate phrases.

        Args:
            candidates (List[List[str]]): List of candidate phrases as lists of words.

        Returns:
            Dict[str, float]: Scores for each candidate phrase.
        """
        # Word frequency and degree (co-occurrence)
        word_freq = Counter()
        word_degree = defaultdict(int)

        for phrase in candidates:
            for word in phrase:
                word_freq[word] += 1
                word_degree[word] += len(phrase) - 1

        # Adjust degree by adding frequency (RAKE standard)
        for word in word_degree:
            word_degree[word] += word_freq[word]

        # Calculate word scores (degree-to-frequency ratio)
        word_score = {word: word_degree[word] / word_freq[word] for word in word_freq}

        # Calculate phrase scores
        phrase_scores = {}
        for phrase in candidates:
            phrase_key = " ".join(phrase)
            phrase_scores[phrase_key] = sum(word_score[word] for word in phrase)

        return phrase_scores

    def _find_adjoining_keywords(self, text: str, keywords: Dict[str, float]) -> Dict[str, float]:
        """
        Identify and score adjoining keywords that contain interior stopwords.

        Args:
            text (str): The input text.
            keywords (Dict[str, float]): Initial keyword scores.

        Returns:
            Dict[str, float]: Updated keyword scores including adjoining keywords.
        """
        text_words = text.lower().split()
        keyword_set = set(keywords.keys())
        adjoining_keywords = {}

        for i in range(len(text_words) - 2):
            word1, word2 = text_words[i], text_words[i + 1]

            # Check if both words are keywords
            if word1 in keyword_set and word2 in keyword_set:
                combined = " ".join(text_words[i:i + 2])

                # Check if this adjoining pair exists multiple times
                count = sum(
                    1 for j in range(len(text_words) - 2)
                    if " ".join(text_words[j:j + 2]) == combined
                )

                if count >= 2:
                    adjoining_keywords[combined] = keywords[word1] + keywords[word2]

        return adjoining_keywords

    def extract_terms_strings(self, corpus: List[str]) -> ATEResults:
        """
        Extract terms from a list of strings using RAKE.

        Args:
            corpus (List[str]): List of input documents.

        Returns:
            ATEResults: Results with terms and scores.
        """
        terms_by_doc = []
        processed_corpus = []

        for idx, doc_text in enumerate(corpus):
            doc_id = f"doc_{idx}"
            processed_corpus.append({"doc_id": doc_id, "text": doc_text})

            # Step 1: Extract and score candidates
            candidates = self._extract_candidates(doc_text)
            phrase_scores = self._score_candidates(candidates)

            # Step 2: Find adjoining keywords
            linked_keywords = self._find_adjoining_keywords(doc_text, phrase_scores)

            # Combine original and linked keywords
            combined_keywords = {**phrase_scores, **linked_keywords}

            # Step 3: Filter by threshold
            terms = [
                {"term": term, "score": score}
                for term, score in combined_keywords.items()
                if self.threshold is None or score >= self.threshold
            ]
            terms_by_doc.append({"doc_id": doc_id, "terms": terms})

        return ATEResults(corpus=processed_corpus, terms=terms_by_doc)

    def extract_terms_teanga(self, corpus) -> ATEResults:
        """
        Extract terms from a Teanga corpus using RAKE.

        Args:
            corpus: A Teanga Corpus object.

        Returns:
            ATEResults: Results with terms, scores, and offsets.
        """
        terms_by_doc = []

        for doc_id in corpus.doc_ids:
            doc = corpus.doc_by_id(doc_id)
            text = doc.text

            # Step 1: Extract and score candidates
            candidates = self._extract_candidates(text)
            phrase_scores = self._score_candidates(candidates)

            # Step 2: Find adjoining keywords
            linked_keywords = self._find_adjoining_keywords(text, phrase_scores)

            # Combine original and linked keywords
            combined_keywords = {**phrase_scores, **linked_keywords}

            # Step 3: Filter by threshold and calculate offsets
            terms = []
            for term, score in combined_keywords.items():
                if self.threshold is None or score >= self.threshold:
                    offsets = []
                    start = text.lower().find(term)
                    while start != -1:
                        end = start + len(term)
                        offsets.append((start, end))
                        start = text.lower().find(term, end)
                    terms.append({"term": term, "score": score, "offsets": offsets})
            terms_by_doc.append({"doc_id": doc_id, "terms": terms})

        return ATEResults(corpus=corpus, terms=terms_by_doc)

