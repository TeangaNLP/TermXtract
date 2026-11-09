import re
from abc import abstractmethod
from collections import Counter

from teanga import Corpus

from ..utils import ATEResults
from .base_extractor import BaseTermExtractor


class NestedTermExtractor(BaseTermExtractor):
    """
    Base class for extractors that score candidates by their frequency and by how they
    are nested in other candidates: C-value, Basic and ComboBasic.

    Subclasses implement `score_terms`. Frequencies and nesting are counted over the whole
    corpus, so a term has the same score in every document it occurs in. One candidate
    is nested in another when its words are a contiguous part of the other's words, so
    "heart" is nested in "heart failure" but not in "hearth". Only candidates, which
    pass the filters of `BaseTermExtractor`, are counted as containing others, and only
    their occurrences that pass the filters are counted.
    """

    @abstractmethod
    def score_terms(
        self, frequencies: Counter[str], containing: dict[str, set[str]]
    ) -> dict[str, float]:
        """
        Score the candidates of the corpus.

        Args:
            frequencies (Counter[str]): The number of occurrences of each candidate in
                the corpus.
            containing (Dict[str, Set[str]]): For each candidate, the other candidates
                it is nested in.

        Returns:
            Dict[str, float]: The score of each candidate.
        """

    def nesting(self, candidates: set[str]) -> dict[str, set[str]]:
        """
        Find the candidates that each candidate is nested in.

        Args:
            candidates (Set[str]): The candidates, with words separated by spaces.

        Returns:
            Dict[str, Set[str]]: For each candidate, the longer candidates that contain
            its words as a contiguous sequence.
        """
        containing: dict[str, set[str]] = {term: set() for term in candidates}
        for term in candidates:
            words = term.split()
            for length in range(1, len(words)):
                for i in range(len(words) - length + 1):
                    part = " ".join(words[i : i + length])
                    if part in containing:
                        containing[part].add(term)
        return containing

    def corpus_scores(self, corpus_ngrams: list[list[str]]) -> dict[str, float]:
        """
        Count the candidates of a corpus and score them with `score_terms`.

        Args:
            corpus_ngrams (List[List[str]]): The occurrences of candidates in each
                document, from `candidate_ngrams_teanga` or `candidate_ngrams_strings`.

        Returns:
            Dict[str, float]: The score of each candidate.
        """
        frequencies = Counter(ngram for doc in corpus_ngrams for ngram in doc)
        return self.score_terms(frequencies, self.nesting(set(frequencies)))

    def extract_terms_teanga(self, corpus: Corpus) -> ATEResults:
        """
        Extract terms from a Teanga corpus.

        Args:
            corpus: A Teanga Corpus object.

        Returns:
            ATEResults: Results with ranked terms and scores, and term offsets.
        """
        ngrams_by_doc = self.candidate_ngrams_teanga(corpus)
        scores = self.corpus_scores(
            [[ngram for ngram, _ in ngrams] for ngrams in ngrams_by_doc.values()]
        )

        terms_by_doc = []
        offsets_by_doc = []
        for doc_id, ngrams_with_offsets in ngrams_by_doc.items():
            doc_terms = self.doc_terms(
                doc_id,
                {
                    ngram: scores[ngram]
                    for ngram, _ in ngrams_with_offsets
                    if ngram in scores
                },
            )
            terms_by_doc.append(doc_terms)
            offsets_by_doc.append(
                self.doc_offsets(doc_id, ngrams_with_offsets, doc_terms)
            )

        return ATEResults(corpus=corpus, terms=terms_by_doc, offsets=offsets_by_doc)

    def extract_terms_strings(self, corpus: list[str]) -> ATEResults:
        """
        Extract terms from a plain list of strings.

        Args:
            corpus (List[str]): List of documents as strings.

        Returns:
            ATEResults: Results with ranked terms and scores.
        """
        corpus_ngrams = self.candidate_ngrams_strings(
            [re.findall(r"\b\w+\b", doc.lower()) for doc in corpus]
        )
        scores = self.corpus_scores(corpus_ngrams)

        terms_by_doc = [
            self.doc_terms(
                f"doc_{idx}",
                {ngram: scores[ngram] for ngram in ngrams if ngram in scores},
            )
            for idx, ngrams in enumerate(corpus_ngrams)
        ]
        processed_corpus = [
            {"doc_id": f"doc_{idx}", "text": doc} for idx, doc in enumerate(corpus)
        ]
        return ATEResults(corpus=processed_corpus, terms=terms_by_doc)
