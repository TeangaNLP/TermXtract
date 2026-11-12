import math
from collections import Counter

from ..stopwords import Stopwords
from .basic import BasicTermExtractor
from .context_extractor import ContextTermExtractor


class DomainCoherenceTermExtractor(ContextTermExtractor):
    r"""
    Domain coherence (Bordea, Buitelaar & Polajnar, 2013), which scores candidates by
    how strongly they are associated with words that characterise the domain.

    1. The `seed_terms` best candidates by Basic are taken as a model of the domain.
    2. The context words are the content words that occur in at least a quarter of the
       documents. The `context_words` of them with the highest average normalised PMI
       with the seed terms are kept.
    3. Each candidate is scored by its average normalised PMI with those context words.

    Normalised PMI is computed from co-occurrence within `window_size` words on either
    side of each occurrence of a candidate:
    \( \log \frac{c(t, w) N}{f(t) f(w)} / -\log \frac{c(t, w)}{N} \), where
    \( N \) is the number of words of the corpus, or -1 if \( t \) and \( w \)
    never co-occur.

    Content words are those that are not stopwords, that match `word_pattern` and, for
    a Teanga corpus with a part-of-speech layer, whose tag is in
    `termxtract.unsupervised.context_extractor.CONTENT_TAGS`.
    """

    def __init__(
        self,
        threshold: float | None = None,
        n: int = 1,
        window_size: int = 5,
        seed_terms: int = 200,
        context_words: int = 50,
        stopwords: Stopwords = "english",
        words_layer: str | None = None,
        lemma_layer: str | None = None,
        min_frequency: int = 1,
        word_pattern: str | None = "alphanumeric",
        pos_pattern: str | None = None,
        pos_layer: str | None = None,
    ):
        """
        Initialize the DomainCoherence extractor.

        Args:
            threshold (Optional[float]): Minimum score for term inclusion.
            n (int): Maximum n-gram size for candidate terms.
            window_size (int): The number of words on either side of a candidate that
                count as co-occurring with it.
            seed_terms (int): The number of best candidates by Basic used to choose the
                context words.
            context_words (int): The number of context words that candidates are
                scored against.
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
            window_size=window_size,
            stopwords=stopwords,
            words_layer=words_layer,
            lemma_layer=lemma_layer,
            min_frequency=min_frequency,
            word_pattern=word_pattern,
            pos_pattern=pos_pattern,
            pos_layer=pos_layer,
        )
        self.seed_terms = seed_terms
        self.context_words = context_words

    def corpus_scores(
        self,
        docs: list[list[str]],
        occurrences: list[list[tuple[str, int, int]]],
        tags: list[list[str]] | None = None,
    ) -> dict[str, float]:
        """
        Score the candidates of a corpus by domain coherence.

        Args:
            docs (List[List[str]]): The words of each document.
            occurrences (List[List[Tuple[str, int, int]]]): The occurrences of the
                candidates in each document, from `candidate_occurrences`.
            tags (Optional[List[List[str]]]): The part-of-speech tag of each word of each
                document, to restrict the context words to `CONTENT_TAGS`, or None.

        Returns:
            Dict[str, float]: The score of each candidate.
        """
        content = self.content_words(docs, tags)
        corpus_size = sum(len(words) for words in docs)
        word_frequencies: Counter[str] = Counter()
        document_frequencies: Counter[str] = Counter()
        for words, is_content in zip(docs, content):
            found = [word for word, ok in zip(words, is_content) if ok]
            word_frequencies.update(found)
            document_frequencies.update(set(found))
        candidate_words = {
            word for word, df in document_frequencies.items() if df >= 0.25 * len(docs)
        }
        term_frequencies = Counter(
            ngram for doc_occurrences in occurrences for ngram, _, _ in doc_occurrences
        )

        def npmi(term: str, word: str, together: int) -> float:
            if together == 0:
                return -1.0
            joint = together / corpus_size
            if joint >= 1:
                return 1.0
            pmi = math.log(
                together
                * corpus_size
                / (term_frequencies[term] * word_frequencies[word])
            )
            return pmi / -math.log(joint)

        # Step 1: the seed terms, the best candidates by Basic
        basic = BasicTermExtractor(n=self.n)
        basic_scores = basic.corpus_scores(
            [
                [ngram for ngram, _, _ in doc_occurrences]
                for doc_occurrences in occurrences
            ]
        )
        seeds = sorted(basic_scores, key=lambda term: (-basic_scores[term], term))[
            : self.seed_terms
        ]
        if not seeds:
            return {}

        # Step 2: the context words most associated with the seed terms
        seed_contexts = self.cooccurrences(
            docs, occurrences, content, set(seeds), candidate_words
        )
        association = {
            word: sum(npmi(seed, word, seed_contexts[seed][word]) for seed in seeds)
            / len(seeds)
            for word in candidate_words
        }
        context = sorted(association, key=lambda word: (-association[word], word))[
            : self.context_words
        ]
        if not context:
            return dict.fromkeys(term_frequencies, 0.0)

        # Step 3: every candidate's average association with the context words
        contexts = self.cooccurrences(docs, occurrences, content, None, set(context))
        return {
            term: sum(npmi(term, word, contexts[term][word]) for word in context)
            / len(context)
            for term in term_frequencies
        }
