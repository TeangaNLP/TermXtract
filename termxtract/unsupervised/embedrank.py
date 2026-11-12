import statistics
from bisect import bisect_right
from collections import defaultdict

import numpy as np
from teanga import Corpus

from ..candidates import WORD
from ..embeddings import DEFAULT_MODEL, Encoder
from ..layers import pos_tags, words_with_offsets
from ..stopwords import Stopwords
from ..utils import ATEResults
from .base_extractor import BaseTermExtractor

VARIANTS = ("embedrank", "mderank")

CORPUS_AGGREGATIONS = ("max", "mean", "centroid")


def normalize(vector: np.ndarray) -> np.ndarray:
    """Scale a vector to unit length, leaving a zero vector as it is."""
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def marginal_relevance(
    relevance: np.ndarray, vectors: np.ndarray, diversity: float
) -> np.ndarray:
    r"""
    Re-rank items by Maximal Marginal Relevance (Carbonell & Goldstein, 1998).

    The items are taken greedily, each time the one with the highest
    \( (1 - \lambda) \cdot relevance - \lambda \cdot redundancy \), where
    \( \lambda \) is `diversity` and the redundancy of an item is its highest cosine
    similarity to the items already taken, or 0 if that is negative.

    Args:
        relevance (np.ndarray): The relevance of each item.
        vectors (np.ndarray): The unit-length embedding of each item, one per row.
        diversity (float): \( \lambda \), from 0, which keeps the order by
            relevance, to 1, which ignores relevance after the first item.

    Returns:
        np.ndarray: The score of each item: the value it was taken at. The values do
        not increase from one item to the next, so they rank the items in the order
        they were taken.
    """
    redundancy = np.zeros(len(relevance))
    taken = np.zeros(len(relevance), dtype=bool)
    scores = np.empty(len(relevance))
    for _ in range(len(relevance)):
        values = (1 - diversity) * relevance - diversity * redundancy
        values[taken] = -np.inf
        best = int(np.argmax(values))
        scores[best] = values[best]
        taken[best] = True
        redundancy = np.maximum(redundancy, vectors @ vectors[best])
    return scores


class EmbedRankTermExtractor(BaseTermExtractor):
    r"""
    Embedding-based ranking: EmbedRank (Bennani-Smires et al., 2018) and MDERank (Zhang
    et al., 2022), which score candidates by how close their meaning is to that of the
    documents they occur in.

    Documents and candidates are embedded by a sentence encoder, by default
    `all-MiniLM-L6-v2` (see `termxtract.embeddings.Encoder`). A document longer than
    the encoder's `max_length` is split into chunks between words, and its embedding
    \( d \) is the mean of the embeddings of its chunks.

    - "embedrank" scores a candidate \( c \) in a document by the cosine similarity of
      its embedding to \( d \).
    - "mderank" masks every occurrence of \( c \) in the document, replacing the tokens
      of its words by the mask token, and re-embeds the chunks it occurs in. The score
      is \( 1 - \cos(d, d_c) \), where \( d_c \) is the embedding of the masked
      document, so candidates whose removal changes the document most score best. It
      encodes each chunk once per candidate in it, so it is much slower than
      "embedrank".

    The scores of a term in the documents it occurs in are combined by
    `corpus_aggregation`. With "centroid", "embedrank" scores each candidate by its
    similarity to the mean of the document embeddings instead.

    The papers rank the noun phrases of a single document, with other encoders: sent2vec
    and doc2vec for EmbedRank, and BERT for MDERank. Here the candidates are those of
    the other methods, set by `n`, `stopwords`, `word_pattern`, `pos_pattern` and
    `min_frequency`. Documents are embedded from their words as written, and candidates
    from their words, or with `lemma_layer` their lemmas.

    With `diversity`, "embedrank" re-ranks the candidates by Maximal Marginal Relevance,
    as in EmbedRank++, so that a candidate similar to better ones ranks lower (see
    `marginal_relevance`). "mderank" does not, since its scores are on another scale
    from the similarities: in long documents, masking a candidate changes few chunks,
    and its scores can all be below 0.03, so the similarities would decide the ranking. The candidates of each document are re-ranked by their scores
    in it, or with a `corpus_aggregation` all candidates together by their combined
    scores. It compares each candidate with every other, so it takes time quadratic in
    their number: about 20 seconds for 16,000 candidates.
    """

    def __init__(
        self,
        threshold: float | None = None,
        n: int = 3,
        variant: str = "embedrank",
        corpus_aggregation: str | None = "max",
        model_name: str = DEFAULT_MODEL,
        encoder: Encoder | None = None,
        diversity: float | None = None,
        stopwords: Stopwords = "english",
        words_layer: str | None = None,
        lemma_layer: str | None = None,
        min_frequency: int = 1,
        word_pattern: str | None = "alphanumeric",
        pos_pattern: str | None = None,
        pos_layer: str | None = None,
    ):
        """
        Initialize the embedding-based extractor.

        Args:
            threshold (Optional[float]): Minimum score for term inclusion.
            n (int): Maximum n-gram size for candidate terms.
            variant (str): "embedrank" or "mderank".
            corpus_aggregation (Optional[str]): How a term's scores in the documents it
                occurs in are combined into one score, which it gets in all of them:
                "max" (the default) for its best score, "mean" for the mean,
                "centroid" for its similarity to the mean of the document embeddings
                ("embedrank" only), or None to keep the score of each document.
            model_name (str): The Hugging Face model of the sentence encoder, if
                `encoder` is None.
            encoder (Optional[Encoder]): A loaded encoder, to share one between
                extractors or set its batch size, maximum length or device, or None to
                load `model_name`.
            diversity (Optional[float]): Re-rank the candidates by Maximal Marginal
                Relevance with this weight on their redundancy, from 0 to 1
                ("embedrank" only), or None to keep the ranking by score.
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
            ValueError: If `variant` or `corpus_aggregation` is unknown, if
                `diversity` is outside 0 to 1 or set for "mderank", if
                `corpus_aggregation` is "centroid" for "mderank", or if the encoder's
                tokenizer has no mask token for "mderank".
            ImportError: If an encoder is to be loaded and the optional 'transformers'
                dependencies are missing.
        """
        if variant not in VARIANTS:
            raise ValueError(
                f"Unknown variant: {variant!r}. Use 'embedrank' or 'mderank'."
            )
        if corpus_aggregation is not None and (
            corpus_aggregation not in CORPUS_AGGREGATIONS
        ):
            raise ValueError(
                f"Unknown corpus_aggregation: {corpus_aggregation!r}. "
                "Use 'max', 'mean', 'centroid' or None."
            )
        if diversity is not None and not 0 <= diversity <= 1:
            raise ValueError(f"diversity must be from 0 to 1, not {diversity}")
        if variant == "mderank" and diversity is not None:
            raise ValueError("diversity needs variant='embedrank'")
        if variant == "mderank" and corpus_aggregation == "centroid":
            raise ValueError("corpus_aggregation='centroid' needs variant='embedrank'")
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
        self.variant = variant
        self.corpus_aggregation = corpus_aggregation
        self.diversity = diversity
        self.encoder = Encoder(model_name) if encoder is None else encoder
        if variant == "mderank" and self.encoder.tokenizer.mask_token_id is None:
            raise ValueError("variant='mderank' needs a tokenizer with a mask token")

    def extract_terms_teanga(self, corpus: Corpus) -> ATEResults:
        """Extract terms from a Teanga corpus."""
        docs = [corpus.doc_by_id(doc_id) for doc_id in corpus.doc_ids]
        words_by_doc = [self.teanga_words(doc) for doc in docs]
        written = [
            [text for _, _, text in words_with_offsets(doc, self.words_layer, None)]
            for doc in docs
        ]
        tags = (
            None
            if self.pos_pattern is None
            else [pos_tags(doc, self.pos_layer, self.words_layer) for doc in docs]
        )
        occurrences = self.candidate_occurrences(
            [[text for _, _, text in words] for words in words_by_doc], tags
        )
        scores_by_doc = self.document_scores(written, occurrences)

        terms_by_doc = []
        offsets_by_doc = []
        for doc_id, words, doc_occurrences, scores in zip(
            corpus.doc_ids, words_by_doc, occurrences, scores_by_doc
        ):
            doc_terms = self.doc_terms(doc_id, scores)
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
        written = [WORD.findall(text) for text in corpus]
        occurrences = self.candidate_occurrences(
            [[word.lower() for word in words] for words in written]
        )
        scores_by_doc = self.document_scores(written, occurrences)
        terms_by_doc = [
            self.doc_terms(f"doc_{idx}", scores)
            for idx, scores in enumerate(scores_by_doc)
        ]
        processed_corpus = [
            {"doc_id": f"doc_{idx}", "text": doc} for idx, doc in enumerate(corpus)
        ]
        return ATEResults(corpus=processed_corpus, terms=terms_by_doc)

    def document_scores(
        self,
        docs: list[list[str]],
        occurrences: list[list[tuple[str, int, int]]],
    ) -> list[dict[str, float]]:
        """
        Score the candidates of each document, combined by `corpus_aggregation` and
        re-ranked by `diversity`.

        Args:
            docs (List[List[str]]): The words of each document as written.
            occurrences (List[List[Tuple[str, int, int]]]): The occurrences of the
                candidates in each document, from `candidate_occurrences`.

        Returns:
            List[Dict[str, float]]: The score of each candidate of each document, where
            higher is better.
        """
        spans = [self.encoder.chunks(words) for words in docs]
        chunk_vectors = self.encoder.encode(
            [words[i:j] for words, doc_spans in zip(docs, spans) for i, j in doc_spans]
        )
        # The embeddings of the chunks of each document
        by_doc = np.split(chunk_vectors, np.cumsum([len(s) for s in spans])[:-1])

        if self.variant == "mderank":
            scores = self.masked_scores(docs, occurrences, spans, by_doc)
        else:
            # A document without words has no candidates, so its zero vector is unused
            doc_vectors = [normalize(vectors.sum(axis=0)) for vectors in by_doc]
            terms = sorted({term for doc in occurrences for term, _, _ in doc})
            term_vectors = dict(
                zip(terms, self.encoder.encode([term.split(" ") for term in terms]))
            )
            if self.corpus_aggregation == "centroid":
                centroid = normalize(
                    np.mean([v for v in doc_vectors if v.any()], axis=0)
                )
                doc_vectors = [centroid] * len(docs)
            scores = [
                {
                    term: float(term_vectors[term] @ doc_vector)
                    for term, _, _ in doc_occurrences
                }
                for doc_occurrences, doc_vector in zip(occurrences, doc_vectors)
            ]

        if self.corpus_aggregation in ("max", "mean"):
            by_term: dict[str, list[float]] = defaultdict(list)
            for doc_scores in scores:
                for term, score in doc_scores.items():
                    by_term[term].append(score)
            combine = max if self.corpus_aggregation == "max" else statistics.fmean
            combined = {term: combine(values) for term, values in by_term.items()}
            scores = [{term: combined[term] for term in doc} for doc in scores]

        if self.diversity is None:
            return scores
        if self.corpus_aggregation is None:
            return [self.diversify(doc_scores) for doc_scores in scores]
        # With a corpus aggregation, a term has the same score in every document, so
        # all the terms are re-ranked together
        reranked = self.diversify(
            {term: score for doc_scores in scores for term, score in doc_scores.items()}
        )
        return [{term: reranked[term] for term in doc_scores} for doc_scores in scores]

    def diversify(self, scores: dict[str, float]) -> dict[str, float]:
        """
        Re-rank candidates by Maximal Marginal Relevance, with `diversity` as the weight
        on their redundancy.

        Args:
            scores (Dict[str, float]): The score of each candidate.

        Returns:
            Dict[str, float]: The score of each candidate after re-ranking.
        """
        assert self.diversity is not None
        terms = list(scores)
        reranked = marginal_relevance(
            np.array([scores[term] for term in terms]),
            self.encoder.encode([term.split(" ") for term in terms]),
            self.diversity,
        )
        return dict(zip(terms, reranked.tolist()))

    def masked_scores(
        self,
        docs: list[list[str]],
        occurrences: list[list[tuple[str, int, int]]],
        spans: list[list[tuple[int, int]]],
        chunk_vectors: list[np.ndarray],
    ) -> list[dict[str, float]]:
        """
        Score the candidates of each document by MDERank.

        Args:
            docs (List[List[str]]): The words of each document as written.
            occurrences (List[List[Tuple[str, int, int]]]): The occurrences of the
                candidates in each document.
            spans (List[List[Tuple[int, int]]]): The chunks of each document, from
                `Encoder.chunks`.
            chunk_vectors (List[np.ndarray]): The embeddings of the chunks of each
                document.

        Returns:
            List[Dict[str, float]]: The score of each candidate of each document.
        """
        # The chunks to encode with each candidate masked: for each document, term
        # and chunk, the indices of the masked words within the chunk
        masked: list[dict[str, dict[int, set[int]]]] = []
        for doc_occurrences, doc_spans in zip(occurrences, spans):
            starts = [i for i, _ in doc_spans]
            doc_masks: dict[str, dict[int, set[int]]] = defaultdict(
                lambda: defaultdict(set)
            )
            for term, i, j in doc_occurrences:
                for word in range(i, j):
                    chunk = bisect_right(starts, word) - 1
                    doc_masks[term][chunk].add(word - starts[chunk])
            masked.append(doc_masks)
        texts = [
            docs[k][spans[k][chunk][0] : spans[k][chunk][1]]
            for k, doc_masks in enumerate(masked)
            for chunks in doc_masks.values()
            for chunk in chunks
        ]
        masks = [
            words
            for doc_masks in masked
            for chunks in doc_masks.values()
            for words in chunks.values()
        ]
        new_vectors = iter(self.encoder.encode(texts, masks))

        scores = []
        for doc_masks, vectors in zip(masked, chunk_vectors):
            total = vectors.sum(axis=0)
            doc_scores = {}
            for term, chunks in doc_masks.items():
                changed = total.copy()
                for chunk in chunks:
                    changed += next(new_vectors) - vectors[chunk]
                doc_scores[term] = 1 - float(normalize(total) @ normalize(changed))
            scores.append(doc_scores)
        return scores
