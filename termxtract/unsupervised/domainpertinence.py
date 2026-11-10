from ..stopwords import Stopwords
from .reference_extractor import ReferenceCorpus, ReferenceCorpusTermExtractor

TIE_BREAKS = ("ratio", "frequency")
"""The ways of ordering the candidates that tie at a Domain Pertinence of 1."""


class DomainPertinenceTermExtractor(ReferenceCorpusTermExtractor):
    r"""
    Domain Pertinence (Navigli & Velardi, 2004), which compares a candidate's frequency
    in the target corpus with its frequency in the reference corpus where it is most
    frequent.

    Each candidate is scored as
    \( p_T(t) / \max(p_T(t), \max_j p_j(t)) \), where \( p_T(t) \) and \( p_j(t) \) are
    its frequencies relative to the number of n-grams in the target corpus and in the
    \( j \)-th reference corpus. Navigli & Velardi use one reference corpus per
    contrasting domain. The score is 1 for every candidate at least as frequent in the
    target corpus as in any reference corpus, including those missing from them all, so
    with a single reference corpus most candidates tie at 1. `tie_break` orders those
    candidates, by adding a number between 0 and 1 to their score:

    - `"ratio"`: by \( p_T(t) / \max_j p_j(t) \), with the reference counts smoothed by
      adding one, as in weirdness. With a single reference corpus, candidates are then
      ranked as weirdness ranks them.
    - `"frequency"`: by their frequency in the target corpus.
    - None: leave them tied, so they are ranked in arbitrary order.
    """

    def __init__(
        self,
        reference_corpus: ReferenceCorpus | list[ReferenceCorpus] | None,
        threshold: float | None = None,
        n: int = 1,
        tie_break: str | None = "ratio",
        stopwords: Stopwords = "english",
        words_layer: str | None = None,
        lemma_layer: str | None = None,
        min_frequency: int = 1,
        word_pattern: str | None = "alphanumeric",
        pos_pattern: str | None = None,
        pos_layer: str | None = None,
    ):
        """
        Initialize the Domain Pertinence extractor.

        Args:
            reference_corpus (Union[Corpus, List[str], List[Union[Corpus, List[str]]]]):
                The reference corpus, of the same type as the corpus the terms are
                extracted from, or a list of reference corpora, such as one per
                contrasting domain.
            threshold (Optional[float]): Minimum score for term inclusion.
            n (int): Maximum n-gram size (e.g., 1 for unigrams, 2 for bigrams, etc.).
            tie_break (Optional[str]): How to order the candidates that score 1:
                "ratio", "frequency" or None (see the class description).
            stopwords (Union[str, Iterable[str], None]): Drop candidates that start or end
                with a stopword: "english", "hindi", a list of words, or None.
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
            ValueError: If `tie_break` is not one of `TIE_BREAKS` or None.
        """
        if tie_break is not None and tie_break not in TIE_BREAKS:
            raise ValueError(
                f"Unknown tie_break: {tie_break!r}. Use one of "
                f"{', '.join(map(repr, TIE_BREAKS))}, or None."
            )
        super().__init__(
            reference_corpus=reference_corpus,
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
        self.tie_break = tie_break

    def score_terms(
        self, target_ngrams: list[list[str]], ref_ngrams: list[list[str]]
    ) -> dict[str, float]:
        """
        Score the n-grams of the target corpus by Domain Pertinence against one
        reference corpus.

        Args:
            target_ngrams (List[List[str]]): The n-grams of each target document.
            ref_ngrams (List[List[str]]): The n-grams of each reference document.

        Returns:
            Dict[str, float]: The Domain Pertinence of every n-gram in the target corpus.
        """
        return self.score_corpora(target_ngrams, [ref_ngrams])

    def score_corpora(
        self, target_ngrams: list[list[str]], ref_ngrams: list[list[list[str]]]
    ) -> dict[str, float]:
        """
        Score the n-grams of the target corpus by Domain Pertinence.

        Args:
            target_ngrams (List[List[str]]): The n-grams of each target document.
            ref_ngrams (List[List[List[str]]]): The n-grams of each document of each
                reference corpus.

        Returns:
            Dict[str, float]: The Domain Pertinence of every n-gram in the target corpus,
            between 0 and 1, plus the tie-breaking value for those at 1.
        """
        target_counts, target_total = self.count_ngrams(target_ngrams)
        references = [self.count_ngrams(corpus) for corpus in ref_ngrams]
        scores = {}
        for ngram, count in target_counts.items():
            p_target = count / target_total
            p_ref = max(counts[ngram] / total for counts, total in references)
            scores[ngram] = p_target / max(p_target, p_ref)
            if scores[ngram] < 1 or self.tie_break is None:
                continue
            if self.tie_break == "ratio":
                value = p_target / max(
                    (counts[ngram] + 1) / total for counts, total in references
                )
            else:
                value = count
            # Map the value, which is positive, to between 0 and 1, keeping its order
            scores[ngram] += value / (1 + value)
        return scores
