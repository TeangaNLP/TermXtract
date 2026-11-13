import re

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from ..stopwords import Stopwords
from .context_extractor import ContextTermExtractor

NODE_TAGS = re.compile(r"NOUN|PROPN|ADJ|NN.*|JJ.*")
"""
The part-of-speech tags of the words that can be nodes of the graph: nouns and
adjectives, as Universal Dependencies or Penn Treebank tags.
"""

WINDOW_SIZES = {"plain": 1, "single": 9, "position": 9}
"""
The default `window_size` of each variant. Words within 1 word of each other co-occur
in a window of 2 words, as in TextRank, and within 9 words in a window of 10, as in
SingleRank and PositionRank.
"""

GRAPHS = ("corpus", "document")

DAMPING = 0.85
"""The probability that the random walk of PageRank follows an edge."""

TOLERANCE = 1e-10
"""PageRank stops when the scores change by less than this in total."""


def pagerank(weights: csr_matrix, personalization: np.ndarray) -> np.ndarray:
    """
    Score the nodes of an undirected graph by PageRank, by power iteration.

    The random walk follows an edge with probability `DAMPING`, choosing it in
    proportion to its weight, and otherwise jumps to a node chosen by `personalization`.
    From a node with no edges, it always jumps.

    Args:
        weights (csr_matrix): The symmetric matrix of the weights of the edges.
        personalization (np.ndarray): The probability of jumping to each node.

    Returns:
        np.ndarray: The score of each node. The scores sum to 1.
    """
    strength = np.asarray(weights.sum(axis=1)).ravel()
    dangling = strength == 0
    inverse = np.divide(1.0, strength, out=np.zeros_like(strength), where=~dangling)
    scores = np.full(len(strength), 1 / len(strength))
    while True:
        previous = scores
        jumps = (1 - DAMPING) + DAMPING * previous[dangling].sum()
        scores = DAMPING * (weights @ (previous * inverse)) + jumps * personalization
        if np.abs(scores - previous).sum() < TOLERANCE:
            return scores


def window_graph(
    docs: list[list[str]], is_node: list[list[bool]], window_size: int
) -> tuple[dict[str, int], np.ndarray, csr_matrix]:
    """
    Build the graph of the words that occur within `window_size` words of each other.

    Windows do not cross documents, and the words that are not nodes take up places in
    them.

    Args:
        docs (List[List[str]]): The words of each document.
        is_node (List[List[bool]]): For each word of each document, whether it is a
            node.
        window_size (int): Link the nodes that occur within this many words of each
            other.

    Returns:
        Tuple[Dict[str, int], np.ndarray, csr_matrix]: The index of each node; the
        node of each place, -1 for the words that are not nodes and for `window_size`
        places after each document; and the symmetric matrix of how often each pair of
        nodes occurs within a window.
    """
    vocabulary: dict[str, int] = {}
    nodes: list[int] = []
    for doc, keep in zip(docs, is_node):
        nodes.extend(
            vocabulary.setdefault(word, len(vocabulary)) if node else -1
            for word, node in zip(doc, keep)
        )
        nodes.extend([-1] * window_size)
    ids = np.array(nodes, dtype=int)
    pairs = [
        (ids[:-distance], ids[distance:]) for distance in range(1, window_size + 1)
    ]
    linked = [(a >= 0) & (b >= 0) & (a != b) for a, b in pairs]
    rows = np.concatenate([a[keep] for (a, _), keep in zip(pairs, linked)])
    cols = np.concatenate([b[keep] for (_, b), keep in zip(pairs, linked)])
    size = len(vocabulary)
    counts = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(size, size)).tocsr()
    return vocabulary, ids, counts + counts.T


class TextRankTermExtractor(ContextTermExtractor):
    r"""
    Graph-based ranking: TextRank (Mihalcea & Tarau, 2004), SingleRank (Wan & Xiao,
    2008) and PositionRank (Florescu & Caragea, 2017).

    The words are the nodes of a graph, in which two words are linked if they occur
    within `window_size` words of each other. Each word is scored by PageRank over the
    graph, with a damping factor of 0.85, and each candidate by the sum of the scores of
    its words. The variants differ in the graph:

    - "plain", TextRank: the edges are unweighted.
    - "single", SingleRank: each edge is weighted by how often its words co-occur.
    - "position", PositionRank: the edges are weighted as in SingleRank, and the random
      walk jumps to each word in proportion to \( \sum 1 / p \) over its occurrences,
      where \( p \) is the position of the occurrence in its document, counting from 1,
      so words that occur early score better.

    The nodes are the words that are not stopwords, that match `word_pattern` and, for
    a Teanga corpus with a part-of-speech layer, whose tag is in `NODE_TAGS`: nouns and
    adjectives. Other words, such as "of" in "degree of freedom", take up places in the
    window but score 0. Windows cross sentences but not documents, and in a Teanga
    corpus each punctuation token of the words layer takes up a place, as in pke.

    By default, one graph is built over the whole corpus, so a term has the same score
    in every document. With `graph="document"`, each document has a graph of its own,
    as in the papers, which rank the keyphrases of a single document.

    The papers rank only the longest sequences of nouns and adjectives (TextRank and
    SingleRank) or noun phrases (PositionRank). Here the candidates are those of the
    other methods, set by `n`, `stopwords`, `word_pattern`, `pos_pattern` and
    `min_frequency`, and TextRank's joining of adjacent top-ranked words is not done.
    Given the same words, nodes and candidates, the scores are those of pke's
    `TextRank`, `SingleRank` and `PositionRank`, run until PageRank converges: pke stops
    it at networkx's tolerance of 1e-4, where the scores of low-ranked words can still
    be several percent off.
    """

    content_tags = NODE_TAGS

    def __init__(
        self,
        threshold: float | None = None,
        n: int = 3,
        variant: str = "plain",
        window_size: int | None = None,
        graph: str = "corpus",
        stopwords: Stopwords = "english",
        words_layer: str | None = None,
        lemma_layer: str | None = None,
        min_frequency: int = 1,
        word_pattern: str | None = "alphanumeric",
        pos_pattern: str | None = None,
        pos_layer: str | None = None,
    ):
        """
        Initialize the graph-based extractor.

        Args:
            threshold (Optional[float]): Minimum score for term inclusion.
            n (int): Maximum n-gram size for candidate terms.
            variant (str): "plain" for TextRank, "single" for SingleRank or "position"
                for PositionRank.
            window_size (Optional[int]): Link the words that occur within this many
                words of each other, or None for the variant's default in
                `WINDOW_SIZES`: 1 for TextRank and 9 for the others.
            graph (str): "corpus" to build one graph over the whole corpus, or
                "document" to build one for each document.
            stopwords (Union[str, Iterable[str], None]): Drop candidates that start or end
                with a stopword, and leave stopwords out of the graph: a name in
                `STOPWORD_LISTS`, a list of words, or None.
            words_layer (Optional[str]): For Teanga input, the span layer holding the
                words, or None for `words` or, failing that, `tokens`.
            lemma_layer (Optional[str]): For Teanga input, a sequence layer holding the
                lemma of each word, to count inflected forms such as "cell" and "cells"
                as one node and one term, or None to use the words as written.
            min_frequency (int): Drop candidates that occur fewer times in the corpus.
            word_pattern (Optional[str]): Drop candidates with a word that does not fully
                match this regular expression or named pattern ("alphanumeric", "atr4s"),
                and leave such words out of the graph, or None to keep every word.
            pos_pattern (Optional[str]): For Teanga input, count only the occurrences
                whose part-of-speech tags match this pattern, or None.
            pos_layer (Optional[str]): The part-of-speech layer for `pos_pattern` and
                for choosing the nodes, or None for `upos` or, failing that, `pos`.

        Raises:
            ValueError: If `variant` or `graph` is unknown, or `window_size` is less
                than 1.
        """
        if variant not in WINDOW_SIZES:
            raise ValueError(
                f"Unknown variant: {variant!r}. Available variants: "
                f"{', '.join(WINDOW_SIZES)}"
            )
        if graph not in GRAPHS:
            raise ValueError(f"Unknown graph: {graph!r}. Use 'corpus' or 'document'.")
        if window_size is None:
            window_size = WINDOW_SIZES[variant]
        if window_size < 1:
            raise ValueError(f"window_size must be at least 1, not {window_size}")
        super().__init__(
            threshold=threshold,
            n=n,
            window_size=window_size,
            stopwords=stopwords,
            words_layer=words_layer,
            lemma_layer=lemma_layer,
            min_frequency=min_frequency,
            word_pattern=word_pattern,
            pos_pattern=pos_pattern,
            pos_layer=pos_layer,
        )
        self.variant = variant
        self.graph = graph

    def document_scores(
        self,
        docs: list[list[str]],
        occurrences: list[list[tuple[str, int, int]]],
        tags: list[list[str]] | None = None,
    ) -> list[dict[str, float]]:
        """
        Score the candidates of each document, over the graph of the corpus or, with
        `graph="document"`, of the document.

        Args:
            docs (List[List[str]]): The words of each document.
            occurrences (List[List[Tuple[str, int, int]]]): The occurrences of the
                candidates in each document, from `candidate_occurrences`.
            tags (Optional[List[List[str]]]): The part-of-speech tag of each word of each
                document, to restrict the nodes to `NODE_TAGS`, or None.

        Returns:
            List[Dict[str, float]]: The score of each candidate of each document.
        """
        if self.graph == "corpus":
            return super().document_scores(docs, occurrences, tags)
        doc_tags: list[list[list[str]] | None] = (
            [None] * len(docs) if tags is None else [[t] for t in tags]
        )
        return [
            self.corpus_scores([doc], [doc_occurrences], t)
            for doc, doc_occurrences, t in zip(docs, occurrences, doc_tags)
        ]

    def corpus_scores(
        self,
        docs: list[list[str]],
        occurrences: list[list[tuple[str, int, int]]],
        tags: list[list[str]] | None = None,
    ) -> dict[str, float]:
        """
        Score the candidates over one graph of the given documents.

        Args:
            docs (List[List[str]]): The words of each document.
            occurrences (List[List[Tuple[str, int, int]]]): The occurrences of the
                candidates in each document, from `candidate_occurrences`.
            tags (Optional[List[List[str]]]): The part-of-speech tag of each word of each
                document, to restrict the nodes to `NODE_TAGS`, or None.

        Returns:
            Dict[str, float]: The score of each candidate.
        """
        is_node = self.content_words(docs, tags)
        vocabulary, ids, weights = window_graph(docs, is_node, self.window_size)
        if not vocabulary:
            return {term: 0.0 for doc in occurrences for term, _, _ in doc}
        size = len(vocabulary)
        if self.variant == "plain":
            weights.data[:] = 1.0

        if self.variant == "position":
            # The inverse position of each occurrence of a node, in the order of ids
            positions = [
                1 / position
                for keep in is_node
                for position, node in enumerate(keep, start=1)
                if node
            ]
            inverse_positions = np.bincount(
                ids[ids >= 0], weights=positions, minlength=size
            )
            personalization = inverse_positions / inverse_positions.sum()
        else:
            personalization = np.full(size, 1 / size)
        word_scores = pagerank(weights, personalization)

        return {
            term: sum(
                word_scores[vocabulary[word]] if word in vocabulary else 0.0
                for word in doc[i:j]
            )
            for doc, doc_occurrences in zip(docs, occurrences)
            for term, i, j in doc_occurrences
        }
