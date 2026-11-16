import re
from collections import Counter
from itertools import pairwise

from teanga import Corpus

from ..utils import ATEResults
from .base_extractor import BaseTermExtractor


class RAKETermExtractor(BaseTermExtractor):
    """
    Rapid Automatic Keyword Extraction (RAKE; Rose et al., 2010).

    The text is split into words at whitespace and at the phrase delimiters, which are
    words of their own, and the candidates are the longest sequences of words that are
    neither stopwords nor phrase delimiters. Each word is scored by its degree, the
    number of words in the candidates it occurs in, divided by its frequency, and each
    candidate by the sum of the scores of its words. Two candidates that are separated
    only by stopwords at least twice in a document, in the same order, are also joined
    into one, such as "axis of evil", scored by the sum of their scores.

    Words are read in lower case, and keep any punctuation that is not a phrase
    delimiter, so the delimiters should include the punctuation of the text.
    """

    def __init__(
        self,
        stoplist: list[str] | None,
        phrase_delimiters: list[str] | None,
        threshold: float | None = None,
    ):
        """
        Initialize the RAKE extractor.

        Args:
            stoplist (List[str]): List of stop words (required).
            phrase_delimiters (List[str]): The strings, such as punctuation, that end a
                phrase wherever they occur, within words or not (required).
            threshold (Optional[float]): Minimum score for term inclusion.
        """
        if stoplist is None or phrase_delimiters is None:
            raise ValueError("RAKE requires both a stoplist and phrase delimiters.")
        # Candidates are already split at the stoplist, so no extra stopword filtering
        super().__init__(threshold=threshold, stopwords=None)
        self.stoplist = {word.lower() for word in stoplist}
        self.phrase_delimiters = set(phrase_delimiters)
        # Longest first, so that a delimiter that contains another is split whole
        self.delimiter_pattern = (
            re.compile(
                "("
                + "|".join(
                    map(
                        re.escape, sorted(self.phrase_delimiters, key=len, reverse=True)
                    )
                )
                + ")"
            )
            if self.phrase_delimiters
            else None
        )

    def tokenize(self, text: str) -> list[str]:
        """
        Split a text into lower-case words, with each phrase delimiter a word.

        Args:
            text (str): Document text.

        Returns:
            List[str]: The words, in order.
        """
        pieces = (
            [text.lower()]
            if self.delimiter_pattern is None
            else self.delimiter_pattern.split(text.lower())
        )
        words = []
        for piece in pieces:
            if piece in self.phrase_delimiters:
                words.append(piece)
            else:
                words.extend(piece.split())
        return words

    def phrases(self, words: list[str]) -> list[tuple[int, int]]:
        """
        Find the candidates in a document's words.

        Args:
            words (List[str]): The words of the document, from `tokenize`.

        Returns:
            List[Tuple[int, int]]: The index of the first word of each candidate and of
            the word after its last, in order.
        """
        spans = []
        start = None
        for i, word in enumerate(words):
            if word in self.stoplist or word in self.phrase_delimiters:
                if start is not None:
                    spans.append((start, i))
                    start = None
            elif start is None:
                start = i
        if start is not None:
            spans.append((start, len(words)))
        return spans

    def generate_candidate_keywords(self, text: str) -> list[list[str]]:
        """
        Generate candidate keywords by splitting on stop words and phrase delimiters.

        Args:
            text (str): Document text.

        Returns:
            List[List[str]]: Candidate keywords as lists of words.
        """
        words = self.tokenize(text)
        return [words[i:j] for i, j in self.phrases(words)]

    def compute_word_scores(self, candidates: list[list[str]]) -> dict[str, float]:
        """
        Score each word by its degree divided by its frequency.

        Args:
            candidates (List[List[str]]): Candidate keywords as lists of words.

        Returns:
            Dict[str, float]: Word scores.
        """
        word_frequency: Counter[str] = Counter()
        word_degree: Counter[str] = Counter()
        for phrase in candidates:
            for word in phrase:
                word_frequency[word] += 1
                # A word co-occurs with every word of its phrase, itself included
                word_degree[word] += len(phrase)

        # Compute degree-to-frequency ratio
        word_scores = {
            word: word_degree[word] / word_frequency[word] for word in word_frequency
        }
        return word_scores

    def compute_candidate_scores(
        self, candidates: list[list[str]], word_scores: dict[str, float]
    ) -> dict[str, float]:
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

    def link_adjoining_keywords(
        self, text: str, candidate_scores: dict[str, float]
    ) -> dict[str, float]:
        """
        Join the candidates that are separated only by stopwords at least twice.

        Args:
            text (str): Document text.
            candidate_scores (Dict[str, float]): Scores for original candidate keywords.

        Returns:
            Dict[str, float]: The candidate scores, with each joined keyword scored by
            the sum of the scores of its two candidates.
        """
        words = self.tokenize(text)
        spans = self.phrases(words)
        linked: Counter[tuple[str, str, str]] = Counter()
        for (i, j), (k, m) in pairwise(spans):
            if all(word in self.stoplist for word in words[j:k]):
                linked[
                    " ".join(words[i:j]), " ".join(words[j:k]), " ".join(words[k:m])
                ] += 1
        scores = dict(candidate_scores)
        for (first, stops, second), count in linked.items():
            if count >= 2:
                scores[f"{first} {stops} {second}"] = (
                    candidate_scores[first] + candidate_scores[second]
                )
        return scores

    def extract_terms_teanga(self, corpus: Corpus) -> ATEResults:
        """
        Extract terms from a Teanga corpus using RAKE.

        Args:
            corpus: A Teanga Corpus object.

        Returns:
            ATEResults: Results with ranked terms and scores, and the character offsets of
            each occurrence of each term.
        """
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

    def extract_terms_strings(self, corpus: list[str]) -> ATEResults:
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
