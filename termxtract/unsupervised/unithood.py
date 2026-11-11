from typing import ClassVar

from teanga import Corpus

from ..association import MEASURES, NgramCounts
from ..candidates import split_words
from ..stopwords import Stopwords
from ..utils import ATEResults
from .base_extractor import BaseTermExtractor

AGGREGATIONS = ("min", "mean")
"""The ways of combining the scores of the splits of a candidate."""


class UnithoodTermExtractor(BaseTermExtractor):
    """
    Base class for extractors that rank multi-word candidates by an association
    (unithood) measure, which scores how strongly their words stick together rather than
    how specific they are to the domain.

    A candidate of k words can be split in k - 1 places into two parts, its first words
    and the rest, and the measure of each split is computed from a contingency table of
    the k-word sequences of the corpus (see `termxtract.association.NgramCounts`). The
    scores of the splits are combined by `aggregation`. Single words have no parts to
    associate, so they are not ranked. The tables count every sequence of words, not
    only the candidates, so the filters of `BaseTermExtractor` change which candidates
    are ranked but not their scores. Scores are computed over the whole corpus, so a
    term has the same score in every document it occurs in.

    Subclasses set `measure` to the name of a measure in
    `termxtract.association.MEASURES`.
    """

    measure: ClassVar[str]

    def __init__(
        self,
        aggregation: str = "min",
        threshold: float | None = None,
        n: int = 2,
        stopwords: Stopwords = "english",
        words_layer: str | None = None,
        lemma_layer: str | None = None,
        min_frequency: int = 1,
        word_pattern: str | None = "alphanumeric",
        pos_pattern: str | None = None,
        pos_layer: str | None = None,
    ):
        """
        Initialize the extractor.

        Args:
            aggregation (str): How to combine the scores of the splits of a candidate
                of three or more words: "min" (the default) scores it by its weakest
                split, and "mean" by the mean of its splits.
            threshold (Optional[float]): Minimum score for term inclusion.
            n (int): Maximum n-gram size, at least 2.
            stopwords (Union[str, Iterable[str], None]): Drop candidates that start or end
                with a stopword: a name in `STOPWORD_LISTS`, a list of words, or None.
            words_layer (Optional[str]): For Teanga input, the span layer holding the
                words, or None for `words` or, failing that, `tokens`.
            lemma_layer (Optional[str]): For Teanga input, a sequence layer holding the
                lemma of each word, to count inflected forms such as "cell" and "cells"
                as one term, or None to use the words as written.
            min_frequency (int): Drop candidates that occur fewer times in the corpus.
            word_pattern (Optional[str]): Drop candidates with a word that does not fully
                match this regular expression or named pattern ("alphanumeric", "atr4s"),
                or None to keep every word.
            pos_pattern (Optional[str]): For Teanga input, count only the occurrences
                whose part-of-speech tags match this pattern, or None.
            pos_layer (Optional[str]): The part-of-speech layer for `pos_pattern`, or
                None for `upos` or, failing that, `pos`.

        Raises:
            ValueError: If `n` is less than 2 or `aggregation` is unknown.
        """
        if n < 2:
            raise ValueError(
                "Unithood measures rank multi-word terms, so n must be at least 2."
            )
        if aggregation not in AGGREGATIONS:
            raise ValueError(
                f"Unknown aggregation: {aggregation!r}. "
                f"Available aggregations: {', '.join(AGGREGATIONS)}"
            )
        self.aggregation = aggregation
        super().__init__(
            threshold=threshold,
            n=n,
            stopwords=stopwords,
            words_layer=words_layer,
            lemma_layer=lemma_layer,
            min_frequency=min_frequency,
            word_pattern=word_pattern,
            pos_pattern=pos_pattern,
            pos_layer=pos_layer,
        )

    def corpus_scores(
        self, docs: list[list[str]], corpus_ngrams: list[list[str]]
    ) -> dict[str, float]:
        """
        Score the multi-word candidates of a corpus.

        Args:
            docs (List[List[str]]): The words of each document.
            corpus_ngrams (List[List[str]]): The occurrences of candidates in each
                document, from `candidate_ngrams_teanga` or `candidate_ngrams_strings`.

        Returns:
            Dict[str, float]: The score of each candidate of two or more words.
        """
        counts = NgramCounts(docs, self.n)
        column = MEASURES.index(self.measure)
        scores = {}
        for candidate in {ngram for doc in corpus_ngrams for ngram in doc}:
            words = tuple(candidate.split(" "))
            if len(words) < 2:
                continue
            splits = [counts.measures(words, i)[column] for i in range(1, len(words))]
            scores[candidate] = (
                min(splits) if self.aggregation == "min" else sum(splits) / len(splits)
            )
        return scores

    def extract_terms_teanga(self, corpus: Corpus) -> ATEResults:
        """
        Extract terms from a Teanga corpus.

        Args:
            corpus: A Teanga Corpus object.

        Returns:
            ATEResults: Results with ranked terms and scores, and term offsets.
        """
        ngrams_by_doc = self.candidate_ngrams_teanga(corpus)
        docs = [
            [text for _, _, text in self.teanga_words(corpus.doc_by_id(doc_id))]
            for doc_id in corpus.doc_ids
        ]
        scores = self.corpus_scores(
            docs, [[ngram for ngram, _ in ngrams] for ngrams in ngrams_by_doc.values()]
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
        docs = [split_words(doc) for doc in corpus]
        corpus_ngrams = self.candidate_ngrams_strings(docs)
        scores = self.corpus_scores(docs, corpus_ngrams)

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


class PMITermExtractor(UnithoodTermExtractor):
    r"""
    Pointwise mutual information (Church and Hanks, 1990): \( \log_2 \frac{P(x y)}{P(x)
    P(y)} \). It favours rare candidates whose words seldom occur apart.
    """

    measure = "MI"


class LLRTermExtractor(UnithoodTermExtractor):
    """Dunning's (1993) log-likelihood ratio of the words occurring together by chance."""

    measure = "L"


class Chi2TermExtractor(UnithoodTermExtractor):
    """Pearson's χ² statistic of the contingency table (Manning and Schütze, 1999)."""

    measure = "Khi2"


class DiceTermExtractor(UnithoodTermExtractor):
    """
    The Dice coefficient (Smadja et al., 1996): twice the frequency of the candidate
    over the sum of the frequencies of its two parts.
    """

    measure = "Dice"


class TScoreTermExtractor(UnithoodTermExtractor):
    """
    Student's t-score of the candidate's frequency against its frequency if its parts
    were independent (Manning and Schütze, 1999).
    """

    measure = "T test"
