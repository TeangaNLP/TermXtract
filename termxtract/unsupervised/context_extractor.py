import re
from abc import abstractmethod
from collections import Counter

from teanga import Corpus

from ..candidates import split_words
from ..layers import POS_LAYERS, pos_tags
from ..stopwords import Stopwords
from ..utils import ATEResults
from .base_extractor import BaseTermExtractor

CONTENT_TAGS = re.compile(r"NOUN|PROPN|ADJ|VERB|NN.*|JJ.*|VB.*")
"""
The part-of-speech tags of the words that can be context words: nouns, adjectives and
verbs, as Universal Dependencies or Penn Treebank tags.
"""


class ContextTermExtractor(BaseTermExtractor):
    """
    Base class for extractors that score candidates by the words around their
    occurrences: Domain coherence and NC-value.

    Subclasses implement `corpus_scores`. The context of an occurrence is the content
    words within `window_size` words on either side of it. Content words are those that
    are not stopwords, that match `word_pattern` and, for a Teanga corpus with a
    part-of-speech layer, whose tag is in `CONTENT_TAGS`.
    """

    def __init__(
        self,
        threshold: float | None = None,
        n: int = 1,
        window_size: int = 5,
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
            threshold (Optional[float]): Minimum score for term inclusion.
            n (int): Maximum n-gram size for candidate terms.
            window_size (int): The number of words on either side of a candidate that
                count as its context.
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
            pos_layer (Optional[str]): The part-of-speech layer for `pos_pattern` and
                for choosing the context words, or None for `upos` or, failing that,
                `pos`.
        """
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
        self.window_size = window_size

    @abstractmethod
    def corpus_scores(
        self,
        docs: list[list[str]],
        occurrences: list[list[tuple[str, int, int]]],
        tags: list[list[str]] | None = None,
    ) -> dict[str, float]:
        """
        Score the candidates of a corpus.

        Args:
            docs (List[List[str]]): The words of each document.
            occurrences (List[List[Tuple[str, int, int]]]): The occurrences of the
                candidates in each document, from `candidate_occurrences`.
            tags (Optional[List[List[str]]]): The part-of-speech tag of each word of each
                document, to restrict the context words to `CONTENT_TAGS`, or None.

        Returns:
            Dict[str, float]: The score of each candidate.
        """

    def extract_terms_teanga(self, corpus: Corpus) -> ATEResults:
        """Extract terms from a Teanga corpus."""
        words_by_doc = [
            self.teanga_words(corpus.doc_by_id(doc_id)) for doc_id in corpus.doc_ids
        ]
        docs = [[text for _, _, text in words] for words in words_by_doc]
        tags = None
        if self.pos_layer is not None or any(
            layer in corpus.meta for layer in POS_LAYERS
        ):
            tags = [
                pos_tags(corpus.doc_by_id(doc_id), self.pos_layer, self.words_layer)
                for doc_id in corpus.doc_ids
            ]
        occurrences = self.candidate_occurrences(
            docs, None if self.pos_pattern is None else tags
        )
        scores = self.corpus_scores(docs, occurrences, tags)

        terms_by_doc = []
        offsets_by_doc = []
        for doc_id, words, doc_occurrences in zip(
            corpus.doc_ids, words_by_doc, occurrences
        ):
            doc_terms = self.doc_terms(
                doc_id, {ngram: scores[ngram] for ngram, _, _ in doc_occurrences}
            )
            terms_by_doc.append(doc_terms)
            offsets_by_doc.append(
                self.doc_offsets(
                    doc_id,
                    (
                        (ngram, (words[i][0], words[j - 1][1]))
                        for ngram, i, j in doc_occurrences
                    ),
                    doc_terms,
                )
            )

        return ATEResults(corpus=corpus, terms=terms_by_doc, offsets=offsets_by_doc)

    def extract_terms_strings(self, corpus: list[str]) -> ATEResults:
        """Extract terms from a plain list of strings."""
        docs = [split_words(doc) for doc in corpus]
        occurrences = self.candidate_occurrences(docs)
        scores = self.corpus_scores(docs, occurrences)

        terms_by_doc = [
            self.doc_terms(
                f"doc_{idx}", {ngram: scores[ngram] for ngram, _, _ in doc_occurrences}
            )
            for idx, doc_occurrences in enumerate(occurrences)
        ]
        processed_corpus = [
            {"doc_id": f"doc_{idx}", "text": doc} for idx, doc in enumerate(corpus)
        ]
        return ATEResults(corpus=processed_corpus, terms=terms_by_doc)

    def content_words(
        self, docs: list[list[str]], tags: list[list[str]] | None = None
    ) -> list[list[bool]]:
        """
        Find the words that can be context words.

        Args:
            docs (List[List[str]]): The words of each document.
            tags (Optional[List[List[str]]]): The part-of-speech tag of each word of each
                document, or None.

        Returns:
            List[List[bool]]: For each word of each document, whether it is a content
            word.
        """
        return [
            [
                word.lower() not in self.stopwords
                and (
                    self.word_pattern is None or bool(self.word_pattern.fullmatch(word))
                )
                and (tags is None or bool(CONTENT_TAGS.fullmatch(tags[k][i])))
                for i, word in enumerate(words)
            ]
            for k, words in enumerate(docs)
        ]

    def cooccurrences(
        self,
        docs: list[list[str]],
        occurrences: list[list[tuple[str, int, int]]],
        content: list[list[bool]],
        terms: set[str] | None,
        words: set[str] | None,
    ) -> dict[str, Counter[str]]:
        """
        Count how often each content word occurs within `window_size` words of each
        occurrence of each candidate.

        Args:
            docs (List[List[str]]): The words of each document.
            occurrences (List[List[Tuple[str, int, int]]]): The occurrences of the
                candidates in each document.
            content (List[List[bool]]): Which words are content words, from
                `content_words`.
            terms (Optional[Set[str]]): The candidates to count for, or None for every
                candidate.
            words (Optional[Set[str]]): The content words to count, or None for every
                content word.

        Returns:
            Dict[str, Counter[str]]: For each candidate, how often each word occurs in
            its context.
        """
        counts: dict[str, Counter[str]] = {}
        for doc, is_content, doc_occurrences in zip(docs, content, occurrences):
            for term, start, end in doc_occurrences:
                if terms is not None and term not in terms:
                    continue
                window = (
                    range(max(0, start - self.window_size), start),
                    range(end, min(len(doc), end + self.window_size)),
                )
                counts.setdefault(term, Counter()).update(
                    doc[i]
                    for side in window
                    for i in side
                    if is_content[i] and (words is None or doc[i] in words)
                )
        return counts
