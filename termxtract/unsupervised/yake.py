import math
import re
import statistics
from collections import Counter, defaultdict

from teanga import Corpus

from ..candidates import WORD
from ..layers import pos_tags
from ..stopwords import Stopwords
from ..utils import ATEResults
from .base_extractor import BaseTermExtractor

SENTENCE_END = re.compile(r"[.!?\u0964\u0965\u3002\uff01\uff1f]")
"""The punctuation that ends a sentence: ".", "!" and "?", the Devanagari danda and
double danda, and the CJK full stop, exclamation mark and question mark."""

CORPUS_AGGREGATIONS = ("max", "mean")


def is_near_duplicate(a: str, b: str, threshold: float) -> bool:
    """
    Check whether two strings are more similar than a threshold.

    The similarity is \\( 1 - d / l \\), where \\( d \\) is the Levenshtein distance and
    \\( l \\) the length of the longer string. Only distances up to the largest one that
    passes the threshold are computed.

    Args:
        a (str): A string.
        b (str): Another string.
        threshold (float): The similarity to exceed.

    Returns:
        bool: Whether the similarity of `a` and `b` is greater than `threshold`.
    """
    longer = max(len(a), len(b))
    # The largest distance d with 1 - d / longer > threshold
    limit = longer
    while limit >= 0 and not 1 - limit / longer > threshold:
        limit -= 1
    if limit < 0 or abs(len(a) - len(b)) > limit:
        return False
    previous = list(range(len(b) + 1))
    for i, char in enumerate(a, start=1):
        current = [i]
        for j, other in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (char != other),
                )
            )
        if min(current) > limit:
            return False
        previous = current
    return previous[-1] <= limit


class YAKETermExtractor(BaseTermExtractor):
    r"""
    YAKE! (Campos et al., 2020), which scores the candidates of each document from local
    features of their words, with no corpus or training data.

    The words are counted case-insensitively. Each word \( t \) of a document gets five
    features:

    - casing, \( T_{case} = \max(TF_U(t), TF_A(t)) / (1 + \ln TF(t)) \), where
      \( TF_A \) counts its occurrences in capitals and \( TF_U \) those whose only
      capital is the first letter, other than at the start of a sentence;
    - position, \( T_{pos} = \ln(\ln(3 + \mathrm{median}(S_t))) \), where \( S_t \) is
      the set of indices of the sentences it occurs in;
    - frequency, \( T_{freq} = TF(t) / (\mu + \sigma) \), where \( \mu \) and
      \( \sigma \) are the mean and standard deviation of the frequencies of the words
      that are not stopwords;
    - relatedness to its context, \( T_{rel} = 1 + (D_L + D_R) \cdot TF(t) /
      \max TF \), where \( \max TF \) is the highest frequency of any word and
      \( D_L \) is the number of different words within `window_size` words on its
      left divided by the number of words there, and \( D_R \) the same on its right.
      Words that don't match `word_pattern` count in no word's context;
    - the share of the sentences it occurs in, \( T_{sent} \).

    Together they score \( S(t) = T_{rel} T_{pos} / (T_{case} + (T_{freq} +
    T_{sent}) / T_{rel}) \), where lower is better. A candidate \( k \) scores
    \( S(k) = \prod_{t \in k} S(t) / (TF(k) (1 + \sum_{t \in k} S(t))) \). A stopword
    inside a candidate, as in "degree of freedom", multiplies the product by
    \( 2 - P \) and lowers the sum by \( 1 - P \), where \( P \) is the probability of
    the word before it being followed by it, times the probability of it being followed
    by the word after it.

    Going down a document's ranking, a candidate is dropped if it is more similar than
    `dedup_threshold` to one already kept, by `is_near_duplicate`. The terms are
    reported in lower case, scored \( -S(k) \) so that higher is better.

    Sentences end at the punctuation of `SENTENCE_END`, such as ".", and a candidate or
    context window does not cross other punctuation either, whether it is between the
    words or a word itself.

    Given the same words, stopwords and lemmas, and a `word_pattern` that rejects the
    numbers and the words mixing letters and digits that `yake` sets aside, the scores
    are those of the `yake` package (version 0.7.3, with `dedup_func="levs"`), except that `lemma_layer` also
    merges the candidates whose words share lemmas, where `yake` merges only the words.
    `yake` reads words ending in "s" as plurals, and words of one or two letters as
    stopwords; use `lemma_layer` and `stopwords` for that here.
    """

    def __init__(
        self,
        threshold: float | None = None,
        n: int = 3,
        window_size: int = 1,
        dedup_threshold: float | None = 0.9,
        corpus_aggregation: str | None = "max",
        stopwords: Stopwords = "english",
        words_layer: str | None = None,
        lemma_layer: str | None = None,
        min_frequency: int = 1,
        word_pattern: str | None = "alphanumeric",
        pos_pattern: str | None = None,
        pos_layer: str | None = None,
    ):
        r"""
        Initialize the YAKE! extractor.

        Args:
            threshold (Optional[float]): Minimum score for term inclusion.
            n (int): Maximum n-gram size for candidate terms.
            window_size (int): The number of words on either side of a word that count
                as its context.
            dedup_threshold (Optional[float]): Drop a candidate that is more similar
                than this to a better one in the same document, or None to keep every
                candidate.
            corpus_aggregation (Optional[str]): How a term's scores in the documents it
                occurs in are combined into one score, which it gets in all of them:
                "max" (the default) for its best score, "mean" for the mean, or None
                to keep the score of each document.
            stopwords (Union[str, Iterable[str], None]): Drop candidates that start or end
                with a stopword: a name in `STOPWORD_LISTS`, a list of words, or None.
            words_layer (Optional[str]): For Teanga input, the span layer holding the
                words, or None for `words` or, failing that, `tokens`.
            lemma_layer (Optional[str]): For Teanga input, a sequence layer holding the
                lemma of each word, to count inflected forms such as "cell" and "cells"
                as one term, or None to use the words as written. The casing feature
                still reads the words as written.
            min_frequency (int): Drop candidates that occur fewer times in the corpus.
            word_pattern (Optional[str]): Drop candidates with a word that does not fully
                match this regular expression or named pattern ("alphanumeric", "atr4s"),
                or None to keep every word.
            pos_pattern (Optional[str]): For Teanga input, count only the occurrences
                whose part-of-speech tags match this pattern, or None.
            pos_layer (Optional[str]): The part-of-speech layer for `pos_pattern`, or
                None for `upos` or, failing that, `pos`.

        Raises:
            ValueError: If `corpus_aggregation` is not "max", "mean" or None.
        """
        if corpus_aggregation is not None and (
            corpus_aggregation not in CORPUS_AGGREGATIONS
        ):
            raise ValueError(
                f"Unknown corpus_aggregation: {corpus_aggregation!r}. "
                "Use 'max', 'mean' or None."
            )
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
        self.dedup_threshold = dedup_threshold
        self.corpus_aggregation = corpus_aggregation

    def extract_terms_teanga(self, corpus: Corpus) -> ATEResults:
        """Extract terms from a Teanga corpus using YAKE!."""
        docs = [corpus.doc_by_id(doc_id) for doc_id in corpus.doc_ids]
        tags = (
            None
            if self.pos_pattern is None
            else [pos_tags(doc, self.pos_layer, self.words_layer) for doc in docs]
        )
        words = [self.teanga_words(doc) for doc in docs]
        return self._extract(
            corpus, list(corpus.doc_ids), [doc.text for doc in docs], words, tags
        )

    def extract_terms_strings(self, corpus: list[str]) -> ATEResults:
        """Extract terms from a plain list of strings using YAKE!."""
        words = [
            [(m.start(), m.end(), m.group().lower()) for m in WORD.finditer(text)]
            for text in corpus
        ]
        processed_corpus = [
            {"doc_id": f"doc_{idx}", "text": doc} for idx, doc in enumerate(corpus)
        ]
        results = self._extract(
            processed_corpus,
            [f"doc_{idx}" for idx in range(len(corpus))],
            corpus,
            words,
            None,
        )
        return ATEResults(corpus=processed_corpus, terms=results.terms)

    def _extract(
        self,
        corpus: Corpus | list[dict[str, str]],
        doc_ids: list[str],
        texts: list[str],
        words: list[list[tuple[int, int, str]]],
        tags: list[list[str]] | None,
    ) -> ATEResults:
        """Score the candidates of each document and build the results."""
        docs = [SplitDocument(text, doc_words) for text, doc_words in zip(texts, words)]
        chunk_occurrences = iter(
            self.candidate_occurrences(
                [[doc.keys[i] for i in chunk] for doc in docs for chunk in doc.chunks],
                None
                if tags is None
                else [
                    [doc_tags[i] for i in chunk]
                    for doc, doc_tags in zip(docs, tags)
                    for chunk in doc.chunks
                ],
            )
        )
        # The occurrences of each document, by its word indices
        occurrences: list[list[tuple[str, int, int]]] = []
        for doc in docs:
            occurrences.append(
                [
                    (ngram, chunk[i], chunk[j - 1] + 1)
                    for chunk in doc.chunks
                    for ngram, i, j in next(chunk_occurrences)
                ]
            )

        doc_scores = [
            self.document_scores(doc, doc_occurrences)
            for doc, doc_occurrences in zip(docs, occurrences)
        ]
        if self.corpus_aggregation is not None:
            by_term: dict[str, list[float]] = defaultdict(list)
            for scores in doc_scores:
                for term, score in scores.items():
                    by_term[term].append(score)
            combine = min if self.corpus_aggregation == "max" else statistics.fmean
            corpus_scores = {term: combine(values) for term, values in by_term.items()}
            doc_scores = [
                {term: corpus_scores[term] for term in scores} for scores in doc_scores
            ]

        terms_by_doc = []
        offsets_by_doc = []
        for doc_id, doc, scores, doc_occurrences in zip(
            doc_ids, docs, doc_scores, occurrences
        ):
            doc_terms = self.doc_terms(
                doc_id, {term: -score for term, score in scores.items()}
            )
            terms_by_doc.append(doc_terms)
            offsets_by_doc.append(
                self.doc_offsets(
                    doc_id,
                    (
                        (ngram, (doc.words[i][0], doc.words[j - 1][1]))
                        for ngram, i, j in doc_occurrences
                    ),
                    doc_terms,
                )
            )
        return ATEResults(corpus=corpus, terms=terms_by_doc, offsets=offsets_by_doc)

    def document_scores(
        self, doc: "SplitDocument", occurrences: list[tuple[str, int, int]]
    ) -> dict[str, float]:
        """
        Score the candidates of one document by YAKE!, where lower is better.

        Args:
            doc (SplitDocument): The document.
            occurrences (List[Tuple[str, int, int]]): The occurrences of the candidates
                in the document, by their start and end (exclusive) word index.

        Returns:
            Dict[str, float]: The YAKE! score of each candidate that is not dropped as
            a near-duplicate.
        """
        frequency: Counter[str] = Counter()
        capitalised: Counter[str] = Counter()
        acronyms: Counter[str] = Counter()
        sentences: dict[str, set[int]] = defaultdict(set)
        left: dict[str, Counter[str]] = defaultdict(Counter)
        right: dict[str, Counter[str]] = defaultdict(Counter)
        pairs: Counter[tuple[str, str]] = Counter()
        # Words that don't match word_pattern take up their place in the windows of
        # other words, but co-occur with none
        usual = [
            self.word_pattern is None or bool(self.word_pattern.fullmatch(key))
            for key in doc.keys
        ]
        for chunk, sentence in zip(doc.chunks, doc.chunk_sentences):
            for index, i in enumerate(chunk):
                key, written = doc.keys[i], doc.written[i]
                frequency[key] += 1
                sentences[key].add(sentence)
                if written.isupper():
                    acronyms[key] += 1
                elif (
                    written[0].isupper()
                    and sum(char.isupper() for char in written) == 1
                    and i not in doc.sentence_starts
                ):
                    capitalised[key] += 1
                if not usual[i]:
                    continue
                for k in chunk[max(0, index - self.window_size) : index]:
                    if not usual[k]:
                        continue
                    left[key][doc.keys[k]] += 1
                    right[doc.keys[k]][key] += 1
                    pairs[doc.keys[k], key] += 1

        valid = [tf for key, tf in frequency.items() if key not in self.stopwords]
        if not valid:
            return {}
        max_frequency = max(frequency.values())
        spread = statistics.fmean(valid) + statistics.pstdev(valid)
        num_sentences = len(doc.sentence_starts)

        def dispersion(context: Counter[str]) -> float:
            total = sum(context.values())
            return len(context) / total if total else 0.0

        word_score = {}
        for key, tf in frequency.items():
            case = max(capitalised[key], acronyms[key]) / (1 + math.log(tf))
            position = math.log(math.log(3 + statistics.median(sentences[key])))
            frequency_norm = tf / spread
            relatedness = 1 + (dispersion(left[key]) + dispersion(right[key])) * (
                tf / max_frequency
            )
            share = len(sentences[key]) / num_sentences
            word_score[key] = (relatedness * position) / (
                case + (frequency_norm + share) / relatedness
            )

        candidate_frequency = Counter(ngram for ngram, _, _ in occurrences)
        scores = {}
        for ngram, start, end in occurrences:
            if ngram in scores:
                continue
            keys = doc.keys[start:end]
            product, total = 1.0, 0.0
            for j, key in enumerate(keys):
                if key in self.stopwords and 0 < j < len(keys) - 1:
                    before, after = keys[j - 1], keys[j + 1]
                    probability = (pairs[before, key] / frequency[before]) * (
                        pairs[key, after] / frequency[after]
                    )
                    product *= 2 - probability
                    total -= 1 - probability
                else:
                    product *= word_score[key]
                    total += word_score[key]
            scores[ngram] = product / ((1 + total) * candidate_frequency[ngram])

        if self.dedup_threshold is None:
            return scores
        kept: dict[str, float] = {}
        for ngram in sorted(scores, key=lambda term: (scores[term], term)):
            if not any(
                is_near_duplicate(ngram, other, self.dedup_threshold) for other in kept
            ):
                kept[ngram] = scores[ngram]
        return kept


class SplitDocument:
    """
    The words of a document, split into sentences and into chunks without punctuation.

    Attributes:
        words (List[Tuple[int, int, str]]): The (start, end, text) of each word.
        keys (List[str]): Each word in lower case, as it is counted.
        written (List[str]): Each word as written in the text.
        chunks (List[List[int]]): The indices of the words of each run of words without
            punctuation between them. Words without a letter or digit are punctuation,
            and are in no chunk.
        chunk_sentences (List[int]): The index of the sentence of each chunk.
        sentence_starts (Set[int]): The index of the first word of each sentence.
    """

    def __init__(self, text: str, words: list[tuple[int, int, str]]):
        self.words = words
        self.keys = [word.lower() for _, _, word in words]
        self.written = [text[start:end] for start, end, _ in words]
        self.chunks: list[list[int]] = []
        self.chunk_sentences: list[int] = []
        self.sentence_starts: set[int] = set()
        sentence = -1
        new_sentence = True
        new_chunk = True
        previous_end = 0
        for i, (start, end, _) in enumerate(words):
            gap = text[previous_end:start]
            previous_end = end
            if SENTENCE_END.search(gap):
                new_sentence = True
            elif gap.strip():
                new_chunk = True
            written = self.written[i]
            if not WORD.search(written):
                if SENTENCE_END.search(written):
                    new_sentence = True
                new_chunk = True
                continue
            if new_sentence:
                sentence += 1
                self.sentence_starts.add(i)
            if new_sentence or new_chunk:
                self.chunks.append([])
                self.chunk_sentences.append(sentence)
            self.chunks[-1].append(i)
            new_sentence = new_chunk = False
