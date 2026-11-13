r"""
Semantic re-ranking of the terms of any extractor: AdaText (Zhang, Petrak & Maynard,
2018) and SemRe-Rank (Zhang, Gao & Ciravegna, 2018).

Both revise the score of each term with the scores of its words in a graph of the
corpus's words, chosen or linked by how similar their embeddings are:

\[ score(t) = \left(1 + \frac{1}{|words(t)|} \sum_{w \in words(t)} s(w)\right)
\cdot ate(t) \]

where \( ate(t) \) is the score of the extractor and \( words(t) \) the distinct words
of the term that are not stopwords. A term with a negative score is divided by the same
factor instead, so that its words' scores move it up the ranking as they do a term with
a positive score. Pass a re-ranker to `TermExtractor` as `rerank`:

    from termxtract import TermExtractor
    from termxtract.rerank import SemReRank

    extractor = TermExtractor(method="cvalue", n=3, rerank=SemReRank(seeds=200))

Words are compared, and terms re-scored, in lower case. Words are embedded one at a time
by a sentence encoder, by default `all-MiniLM-L6-v2` (see
`termxtract.embeddings.Encoder`), so the `transformers` extra is needed.
"""

import math
from abc import ABC, abstractmethod
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix

from .candidates import compile_word_pattern
from .embeddings import DEFAULT_MODEL, Encoder
from .stopwords import Stopwords, load_stopwords
from .unsupervised.textrank import pagerank, window_graph
from .utils import ATEResults, rank_terms

BLOCK = 1024
"""The number of words whose similarities to the others are computed at once."""


class Reranker(ABC):
    """
    Common interface of the re-rankers.

    Subclasses implement `word_scores`, which scores the words of the corpus.
    """

    normalize = False
    """Whether the extractor's scores are divided by their largest absolute value."""

    def __init__(
        self,
        seeds: list[str] | int = 200,
        encoder: Encoder | None = None,
        model_name: str = DEFAULT_MODEL,
        stopwords: Stopwords = "english",
    ):
        r"""
        Initialize the re-ranker.

        Args:
            seeds (Union[List[str], int]): Terms or phrases known to belong to the
                domain, or a number \( z \) to take the \( z \) most frequent terms
                of the extractor, as in the unsupervised variant of SemRe-Rank.
            encoder (Optional[Encoder]): A loaded `termxtract.embeddings.Encoder`, or
                None to load `model_name` when it is first needed.
            model_name (str): The Hugging Face model of the sentence encoder.
            stopwords (Union[str, Iterable[str], None]): The words that are left out of
                the graph and of the words of the terms: a name in `STOPWORD_LISTS`, a
                list of words, or None.

        Raises:
            ValueError: If `seeds` is a number below 1.
        """
        if isinstance(seeds, int) and seeds < 1:
            raise ValueError(f"seeds must be at least 1, not {seeds}")
        self.seeds = seeds
        self.encoder = encoder
        self.model_name = model_name
        self.stopwords = load_stopwords(stopwords)
        self.word_pattern = compile_word_pattern("alphanumeric")

    def rerank(self, results: ATEResults, docs: list[list[str]]) -> ATEResults:
        """
        Re-score the terms of an extractor.

        Args:
            results (ATEResults): The terms of the extractor.
            docs (List[List[str]]): The words of each document of the corpus, as the
                extractor read them, so that the terms are sequences of them.

        Returns:
            ATEResults: The same terms and offsets, with the revised scores.
        """
        docs = [[word.lower() for word in doc] for doc in docs]
        terms = list(
            dict.fromkeys(
                term["term"].lower() for doc in results.terms for term in doc["terms"]
            )
        )
        seeds = (
            self.most_frequent(docs, terms, self.seeds)
            if isinstance(self.seeds, int)
            else [seed.lower() for seed in self.seeds]
        )
        scores = self.word_scores(docs, terms, seeds)
        boost = {}
        for term in terms:
            words = self.words(term)
            boost[term] = (
                1 + sum(scores.get(word, 0.0) for word in words) / len(words)
                if words
                else 1.0
            )
        scale = 1.0
        if self.normalize:
            scale = max(
                (abs(term["score"]) for doc in results.terms for term in doc["terms"]),
                default=0.0,
            )
            scale = scale or 1.0

        def revise(score: float, term: str) -> float:
            score /= scale
            factor = boost[term.lower()]
            return score * factor if score >= 0 else score / factor

        revised = [
            {
                "doc_id": doc["doc_id"],
                "terms": rank_terms(
                    {
                        term["term"]: revise(term["score"], term["term"])
                        for term in doc["terms"]
                    }
                ),
            }
            for doc in results.terms
        ]
        return ATEResults(results.corpus, revised, results.offsets)

    @abstractmethod
    def word_scores(
        self, docs: list[list[str]], terms: list[str], seeds: list[str]
    ) -> dict[str, float]:
        """
        Score the words of the corpus.

        Args:
            docs (List[List[str]]): The lower-case words of each document.
            terms (List[str]): The lower-case terms of the extractor.
            seeds (List[str]): The lower-case seeds.

        Returns:
            Dict[str, float]: The score of each word; the others score 0.
        """

    def is_word(self, word: str) -> bool:
        """Whether a word can be a node: not a stopword, and with a letter or digit."""
        return word not in self.stopwords and bool(self.word_pattern.fullmatch(word))

    def words(self, term: str) -> list[str]:
        """The distinct words of a term or seed that can be nodes."""
        return list(dict.fromkeys(word for word in term.split() if self.is_word(word)))

    @staticmethod
    def most_frequent(docs: list[list[str]], terms: list[str], z: int) -> list[str]:
        """
        Find the most frequent terms in the corpus.

        Args:
            docs (List[List[str]]): The lower-case words of each document.
            terms (List[str]): The lower-case terms.
            z (int): The number of terms to return.

        Returns:
            List[str]: The `z` terms that occur most often, the more frequent first,
            ties in the order of `terms`.
        """
        known = set(terms)
        lengths = {len(term.split()) for term in terms}
        counts: Counter[str] = Counter()
        for doc in docs:
            for n in lengths:
                counts.update(
                    ngram
                    for i in range(len(doc) - n + 1)
                    if (ngram := " ".join(doc[i : i + n])) in known
                )
        order = {term: i for i, term in enumerate(terms)}
        return sorted(terms, key=lambda term: (-counts[term], order[term]))[:z]

    def embed(self, texts: list[list[str]]) -> np.ndarray:
        """Embed texts, loading the encoder if it is not loaded yet."""
        if self.encoder is None:
            self.encoder = Encoder(self.model_name)
        return self.encoder.encode(texts)


class AdaText(Reranker):
    r"""
    AdaText (Zhang, Petrak & Maynard, 2018): TextRank over the words of the corpus
    that are similar to the seeds.

    The nodes are the words of the corpus whose embeddings have a cosine similarity
    above `threshold` with that of any seed, and two nodes are linked if they occur in
    the same window of `window` words in any document, as in TextRank. The score of a
    word \( s(w) \) is its PageRank score, with the scores scaled to unit length, as in
    the TextRank implementation that the paper used (summa), and 0 for the words that
    are not nodes.

    The paper embedded words with GloVe and a seed of several words as the mean of
    their embeddings; here each word, and each seed, is embedded by the sentence
    encoder. Its seeds were a domain lexicon, which need not occur in the corpus.
    """

    def __init__(
        self,
        seeds: list[str] | int = 200,
        threshold: float = 0.7,
        window: int = 5,
        encoder: Encoder | None = None,
        model_name: str = DEFAULT_MODEL,
        stopwords: Stopwords = "english",
    ):
        r"""
        Initialize AdaText.

        Args:
            seeds (Union[List[str], int]): Terms or phrases known to belong to the
                domain, or a number \( z \) to take the \( z \) most frequent terms.
            threshold (float): The cosine similarity to a seed that a word must exceed
                to be a node. The paper tried 0.5 to 0.85 with GloVe embeddings.
            window (int): Link the nodes that occur in the same window of this many
                words: 5 in the paper.
            encoder (Optional[Encoder]): A loaded encoder, or None to load `model_name`.
            model_name (str): The Hugging Face model of the sentence encoder.
            stopwords (Union[str, Iterable[str], None]): The words left out of the
                graph and of the words of the terms.

        Raises:
            ValueError: If `window` is less than 2.
        """
        if window < 2:
            raise ValueError(f"window must be at least 2, not {window}")
        super().__init__(seeds, encoder, model_name, stopwords)
        self.threshold = threshold
        self.window = window

    def word_scores(
        self, docs: list[list[str]], terms: list[str], seeds: list[str]
    ) -> dict[str, float]:
        """Score the words of the corpus by TextRank over the words similar to a seed."""
        vocabulary = list(
            dict.fromkeys(word for doc in docs for word in doc if self.is_word(word))
        )
        if not vocabulary or not seeds:
            return {}
        words = self.embed([[word] for word in vocabulary])
        phrases = self.embed([seed.split() for seed in seeds])
        similar = {
            word
            for start in range(0, len(vocabulary), BLOCK)
            for word, similarity in zip(
                vocabulary[start : start + BLOCK],
                (words[start : start + BLOCK] @ phrases.T).max(axis=1),
            )
            if similarity > self.threshold
        }
        nodes, _, weights = window_graph(
            docs, [[word in similar for word in doc] for doc in docs], self.window - 1
        )
        if not nodes:
            return {}
        weights.data[:] = 1.0
        scores = pagerank(weights, np.full(len(nodes), 1 / len(nodes)))
        scores /= np.linalg.norm(scores)
        return {word: float(scores[i]) for word, i in nodes.items()}


class SemReRank(Reranker):
    r"""
    SemRe-Rank (Zhang, Gao & Ciravegna, 2018): personalised PageRank over graphs of
    semantically related words, one for each document.

    The words are those of the terms. A word \( y \) is strongly related to a word
    \( x \) if the cosine similarity of their embeddings is above `rel_min` and among
    the `rel_top` highest of \( x \) with the other words. Each document has a graph of
    its words, in which each is linked to the words of the document strongly related
    to it, and PageRank, with a damping factor of 0.85, jumps to each word of the seeds
    with the same probability. The semantic importance of a word is the sum of its
    scores over the documents, where a document without a word of the seeds counts 0,
    and \( s(w) \) is its semantic importance divided by the largest. The extractor's
    scores are divided by their largest absolute value too, as in the paper.

    The paper trained word2vec on the corpus; here words are embedded by the sentence
    encoder. In the paper, seeds are the terms that a person picks from the most
    frequent ones; an int `seeds` takes all of them, as in its unsupervised variant. The
    paper's PageRank lets the score of a word without links leak away, where here it
    jumps to the seeds.
    """

    normalize = True

    def __init__(
        self,
        seeds: list[str] | int = 200,
        rel_min: float = 0.5,
        rel_top: float = 0.15,
        encoder: Encoder | None = None,
        model_name: str = DEFAULT_MODEL,
        stopwords: Stopwords = "english",
    ):
        r"""
        Initialize SemRe-Rank.

        Args:
            seeds (Union[List[str], int]): Terms known to belong to the domain, whose
                words are the seeds of PageRank, or a number \( z \) to take the
                \( z \) most frequent terms.
            rel_min (float): The cosine similarity above which words can be strongly
                related: 0.5 in the paper.
            rel_top (float): The fraction of the other words, the most similar, that
                a word can be strongly related to: 0.15 in the paper.
            encoder (Optional[Encoder]): A loaded encoder, or None to load `model_name`.
            model_name (str): The Hugging Face model of the sentence encoder.
            stopwords (Union[str, Iterable[str], None]): The words left out of the
                graph and of the words of the terms.

        Raises:
            ValueError: If `rel_top` is not above 0 and at most 1.
        """
        if not 0 < rel_top <= 1:
            raise ValueError(f"rel_top must be above 0 and at most 1, not {rel_top}")
        super().__init__(seeds, encoder, model_name, stopwords)
        self.rel_min = rel_min
        self.rel_top = rel_top

    def word_scores(
        self, docs: list[list[str]], terms: list[str], seeds: list[str]
    ) -> dict[str, float]:
        """Score the words of the terms by their semantic importance."""
        vocabulary = list(
            dict.fromkeys(word for term in terms for word in self.words(term))
        )
        seed_words = {word for seed in seeds for word in self.words(seed)}
        if len(vocabulary) < 2 or not seed_words:
            return {}
        index = {word: i for i, word in enumerate(vocabulary)}
        vectors = self.embed([[word] for word in vocabulary])
        cutoffs = self.cutoffs(vectors)

        importance = np.zeros(len(vocabulary))
        for doc in docs:
            nodes = np.array(
                list(dict.fromkeys(index[word] for word in doc if word in index)),
                dtype=int,
            )
            seeded = np.array([vocabulary[i] in seed_words for i in nodes], dtype=float)
            if not seeded.any():
                continue
            similarities = vectors[nodes] @ vectors[nodes].T
            related = (similarities >= cutoffs[nodes, None]) & (
                similarities > self.rel_min
            )
            np.fill_diagonal(related, False)
            graph = csr_matrix((related | related.T).astype(float))
            activated = seeded.sum()
            importance[nodes] += pagerank(graph, seeded / activated) * activated
        if not importance.any():
            return {}
        importance /= importance.max()
        return {word: float(importance[i]) for i, word in enumerate(vocabulary)}

    def cutoffs(self, vectors: np.ndarray) -> np.ndarray:
        """
        Find, for each word, the similarity of the least similar of its `rel_top`
        most similar other words.

        Args:
            vectors (np.ndarray): The unit-length embedding of each word, one per row.

        Returns:
            np.ndarray: The cutoff of each word.
        """
        k = max(1, math.ceil(self.rel_top * (len(vectors) - 1)))
        # Each row includes the word itself, at -inf, so the kth highest of the
        # others is at index len - k in ascending order
        kth = len(vectors) - k
        cutoffs = np.empty(len(vectors))
        for start in range(0, len(vectors), BLOCK):
            similarities = vectors[start : start + BLOCK] @ vectors.T
            rows = np.arange(len(similarities))
            similarities[rows, start + rows] = -np.inf
            cutoffs[start : start + BLOCK] = np.partition(similarities, kth, axis=1)[
                :, kth
            ]
        return cutoffs
