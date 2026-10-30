import re
from abc import abstractmethod
from collections import Counter

from teanga import Corpus

from ..utils import ATEResults
from .base_extractor import BaseTermExtractor


class ReferenceCorpusTermExtractor(BaseTermExtractor):
    """
    Base class for extractors that compare the target corpus with a reference corpus.

    Subclasses implement `score_terms`, which scores every n-gram of the target corpus
    from the n-grams of both corpora. Scores are computed over the whole corpus, so a
    term has the same score in every document it occurs in.

    The reference corpus must have the same type as the corpus passed to `extract`: a
    list of strings for plain-text input, or a Teanga `Corpus` with a `words` layer for
    Teanga input.
    """

    def __init__(
        self,
        reference_corpus: Corpus | list[str] | None,
        threshold: float | None = None,
        n: int = 1,
    ):
        """
        Initialize the extractor.

        Args:
            reference_corpus (Union[Corpus, List[str]]): General-language reference corpus,
                of the same type as the corpus the terms are extracted from.
            threshold (Optional[float]): Minimum score for term inclusion.
            n (int): Maximum n-gram size (e.g., 1 for unigrams, 2 for bigrams, etc.).
        """
        super().__init__(threshold=threshold, n=n)
        if reference_corpus is None:
            raise ValueError(f"{type(self).__name__} requires a reference corpus.")
        self.reference_corpus = reference_corpus

    @abstractmethod
    def score_terms(
        self, target_ngrams: list[list[str]], ref_ngrams: list[list[str]]
    ) -> dict[str, float]:
        """
        Score the n-grams of the target corpus.

        Args:
            target_ngrams (List[List[str]]): The n-grams of each target document.
            ref_ngrams (List[List[str]]): The n-grams of each reference document.

        Returns:
            Dict[str, float]: The score of every n-gram in the target corpus.
        """

    @staticmethod
    def count_ngrams(corpus_ngrams: list[list[str]]) -> tuple[Counter, int]:
        """
        Count the n-grams of a corpus.

        Args:
            corpus_ngrams (List[List[str]]): The n-grams of each document.

        Returns:
            Tuple[Counter, int]: The count of each n-gram, and the total number of n-grams.
        """
        counts = Counter(ngram for doc in corpus_ngrams for ngram in doc)
        return counts, sum(counts.values())

    def extract_terms_strings(self, corpus: list[str]) -> ATEResults:
        """
        Extract terms from a list of strings.

        Args:
            corpus (List[str]): The documents as plain strings. The reference corpus must
                also be a list of strings.

        Returns:
            ATEResults: Results with ranked terms and scores.

        Raises:
            TypeError: If the reference corpus is a Teanga corpus.
        """
        if isinstance(self.reference_corpus, Corpus):
            raise TypeError(
                "The reference corpus is a Teanga Corpus, so the corpus to extract "
                "terms from must also be a Teanga Corpus, not a list of strings."
            )
        target_ngrams = [
            self.generate_ngrams_strings(re.findall(r"\b\w+\b", doc.lower()))
            for doc in corpus
        ]
        ref_ngrams = [
            self.generate_ngrams_strings(re.findall(r"\b\w+\b", doc.lower()))
            for doc in self.reference_corpus
        ]
        if not any(ref_ngrams):
            raise ValueError("The reference corpus contains no words.")
        scores = self.score_terms(target_ngrams, ref_ngrams)

        terms_by_doc = [
            self.doc_terms(f"doc_{idx}", {ngram: scores[ngram] for ngram in ngrams})
            for idx, ngrams in enumerate(target_ngrams)
        ]
        return ATEResults(corpus=corpus, terms=terms_by_doc)

    def extract_terms_teanga(self, corpus: Corpus) -> ATEResults:
        """
        Extract terms from a Teanga corpus.

        Args:
            corpus (Corpus): A Teanga corpus with a `words` span layer. The reference corpus
                must also be a Teanga corpus with a `words` layer.

        Returns:
            ATEResults: Results with ranked terms and scores, and term offsets.

        Raises:
            TypeError: If the reference corpus is a list of strings.
        """
        if not isinstance(self.reference_corpus, Corpus):
            raise TypeError(
                "The reference corpus is a list of strings, so the corpus to extract "
                "terms from must also be a list of strings, not a Teanga Corpus."
            )
        ngrams_by_doc = {
            doc_id: self.generate_ngrams_teanga(
                self.teanga_words(corpus.doc_by_id(doc_id))
            )
            for doc_id in corpus.doc_ids
        }
        target_ngrams = [
            [ngram for ngram, _ in ngrams_with_offsets]
            for ngrams_with_offsets in ngrams_by_doc.values()
        ]
        ref_ngrams = [
            [
                ngram
                for ngram, _ in self.generate_ngrams_teanga(
                    self.teanga_words(self.reference_corpus.doc_by_id(doc_id))
                )
            ]
            for doc_id in self.reference_corpus.doc_ids
        ]
        if not any(ref_ngrams):
            raise ValueError("The reference corpus contains no words.")
        scores = self.score_terms(target_ngrams, ref_ngrams)

        terms_by_doc = []
        offsets_by_doc = []
        for doc_id, ngrams_with_offsets in ngrams_by_doc.items():
            doc_terms = self.doc_terms(
                doc_id, {ngram: scores[ngram] for ngram, _ in ngrams_with_offsets}
            )
            terms_by_doc.append(doc_terms)
            offsets_by_doc.append(
                self.doc_offsets(doc_id, ngrams_with_offsets, doc_terms)
            )

        return ATEResults(corpus=corpus, terms=terms_by_doc, offsets=offsets_by_doc)
