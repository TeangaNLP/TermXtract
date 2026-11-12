from collections import Counter

from ..stopwords import Stopwords
from .context_extractor import ContextTermExtractor
from .cvalue import CValueTermExtractor


class NCValueTermExtractor(ContextTermExtractor):
    r"""
    NC-value term extraction (Frantzi, Ananiadou & Mima, 2000), which re-ranks the
    candidates of C-value by the words around them.

    1. The `top_terms` best candidates by C-value are taken as a sample of terms.
    2. Each content word \( b \) within `window_size` words of them is a context word,
       weighted by \( weight(b) = t(b) / n \), where \( t(b) \) is the number of sample
       terms it occurs with and \( n \) is the number of sample terms.
    3. Each candidate \( a \) is scored
       \( \alpha \cdot \text{C-value}(a) + (1 - \alpha) \sum_{b \in C_a} f_a(b)
       \cdot weight(b) \), where \( C_a \) is the set of context words of \( a \),
       \( f_a(b) \) is how often \( b \) occurs within `window_size` words of \( a \),
       and \( \alpha \) is `cvalue_weight`, 0.8 in the paper.

    As in `CValueTermExtractor`, single words are ranked too. Content words are those
    that are not stopwords, that match `word_pattern` and, for a Teanga corpus with a
    part-of-speech layer, whose tag is in
    `termxtract.unsupervised.context_extractor.CONTENT_TAGS`: nouns, adjectives and
    verbs.
    """

    def __init__(
        self,
        threshold: float | None = None,
        n: int = 1,
        window_size: int = 5,
        top_terms: int = 200,
        cvalue_weight: float = 0.8,
        stopwords: Stopwords = "english",
        words_layer: str | None = None,
        lemma_layer: str | None = None,
        min_frequency: int = 1,
        word_pattern: str | None = "alphanumeric",
        pos_pattern: str | None = None,
        pos_layer: str | None = None,
    ):
        r"""
        Initialize the NC-value extractor.

        Args:
            threshold (Optional[float]): Minimum score for term inclusion.
            n (int): Maximum n-gram size for candidate terms.
            window_size (int): The number of words on either side of a candidate that
                count as its context.
            top_terms (int): The number of best candidates by C-value whose context
                words are weighted.
            cvalue_weight (float): The weight of the C-value, \( \alpha \); the
                context factor has weight \( 1 - \alpha \).
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
        self.top_terms = top_terms
        self.cvalue_weight = cvalue_weight

    def corpus_scores(
        self,
        docs: list[list[str]],
        occurrences: list[list[tuple[str, int, int]]],
        tags: list[list[str]] | None = None,
    ) -> dict[str, float]:
        """
        Score the candidates of a corpus by NC-value.

        Args:
            docs (List[List[str]]): The words of each document.
            occurrences (List[List[Tuple[str, int, int]]]): The occurrences of the
                candidates in each document, from `candidate_occurrences`.
            tags (Optional[List[List[str]]]): The part-of-speech tag of each word of each
                document, to restrict the context words to `CONTENT_TAGS`, or None.

        Returns:
            Dict[str, float]: The score of each candidate.
        """
        cvalue = CValueTermExtractor(n=self.n).corpus_scores(
            [[ngram for ngram, _, _ in doc] for doc in occurrences]
        )
        top = sorted(cvalue, key=lambda term: (-cvalue[term], term))[: self.top_terms]
        if not top:
            return {}

        # Weight each context word by the share of the top terms it occurs with
        content = self.content_words(docs, tags)
        contexts = self.cooccurrences(docs, occurrences, content, None, None)
        with_top: Counter[str] = Counter(
            word for term in top for word in contexts.get(term, Counter())
        )
        weight = {word: count / len(top) for word, count in with_top.items()}

        return {
            term: self.cvalue_weight * score
            + (1 - self.cvalue_weight)
            * sum(
                count * weight.get(word, 0.0)
                for word, count in contexts.get(term, Counter()).items()
            )
            for term, score in cvalue.items()
        }
