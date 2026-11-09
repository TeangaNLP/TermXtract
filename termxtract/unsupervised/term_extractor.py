from typing import Any, ClassVar

from teanga import Corpus

from ..pos import compile_pos_pattern, filter_by_pos
from ..stopwords import Stopwords
from ..utils import ATEResults
from .base_extractor import BaseTermExtractor
from .basic import BasicTermExtractor
from .combobasic import ComboBasicTermExtractor
from .cvalue import CValueTermExtractor
from .domaincoherence import DomainCoherenceTermExtractor
from .domainpertinence import DomainPertinenceTermExtractor
from .ldaextractor import TopicModelingTermExtractor
from .nmfextractor import NMFTermExtractor
from .rake import RAKETermExtractor
from .relevance import RelevanceTermExtractor
from .ridf import RIDFTermExtractor
from .tfidf import TFIDFTermExtractor
from .weirdness import WeirdnessTermExtractor


class TermExtractor:
    """
    A wrapper class for selecting the term extraction method.

    `TermExtractor.METHODS` maps each method name to its extractor class and the
    parameters passed to it, so `list(TermExtractor.METHODS)` lists the available methods.
    """

    METHODS: ClassVar[dict[str, tuple[type[BaseTermExtractor], tuple[str, ...]]]] = {
        "tfidf": (
            TFIDFTermExtractor,
            ("threshold", "n", "stopwords", "words_layer", "lemma_layer"),
        ),
        "ridf": (
            RIDFTermExtractor,
            ("threshold", "n", "stopwords", "words_layer", "lemma_layer"),
        ),
        "cvalue": (
            CValueTermExtractor,
            ("threshold", "n", "stopwords", "words_layer", "lemma_layer"),
        ),
        "basic": (
            BasicTermExtractor,
            ("alpha", "threshold", "n", "stopwords", "words_layer", "lemma_layer"),
        ),
        "combobasic": (
            ComboBasicTermExtractor,
            (
                "alpha",
                "beta",
                "threshold",
                "n",
                "stopwords",
                "words_layer",
                "lemma_layer",
            ),
        ),
        "domaincoherence": (
            DomainCoherenceTermExtractor,
            (
                "window_size",
                "threshold",
                "n",
                "stopwords",
                "words_layer",
                "lemma_layer",
            ),
        ),
        "relevance": (
            RelevanceTermExtractor,
            (
                "reference_corpus",
                "threshold",
                "n",
                "stopwords",
                "words_layer",
                "lemma_layer",
            ),
        ),
        "topicmodeling": (
            TopicModelingTermExtractor,
            ("num_topics", "threshold", "n", "stopwords", "words_layer", "lemma_layer"),
        ),
        "nmf": (
            NMFTermExtractor,
            (
                "threshold",
                "n_topics",
                "max_features",
                "n",
                "stopwords",
                "words_layer",
                "lemma_layer",
            ),
        ),
        "rake": (RAKETermExtractor, ("stoplist", "phrase_delimiters", "threshold")),
        "domainpertinence": (
            DomainPertinenceTermExtractor,
            (
                "reference_corpus",
                "threshold",
                "n",
                "stopwords",
                "words_layer",
                "lemma_layer",
            ),
        ),
        "weirdness": (
            WeirdnessTermExtractor,
            (
                "reference_corpus",
                "threshold",
                "n",
                "stopwords",
                "words_layer",
                "lemma_layer",
            ),
        ),
    }

    FILTER_PARAMS: ClassVar[tuple[str, ...]] = ("pos_pattern", "pos_layer")
    """Parameters applied to the results by the wrapper, not passed to the extractor."""

    def __init__(
        self,
        method: str = "tfidf",
        alpha: float = 0.5,
        beta: float = 0.5,
        threshold: float | None = None,
        n: int = 1,
        num_topics: int = 20,
        window_size: int = 5,
        stoplist: list[str] | None = None,
        phrase_delimiters: list[str] | None = None,
        reference_corpus: Corpus | list[str] | None = None,
        n_topics: int = 10,
        max_features: int | None = None,
        stopwords: Stopwords = "english",
        words_layer: str | None = None,
        lemma_layer: str | None = None,
        pos_pattern: str | None = None,
        pos_layer: str | None = None,
    ):
        r"""
        Initialize the TermExtractor with the desired method and parameters.

        Each method only uses some of the parameters; the others are ignored.

        Args:
            method (str): The term extraction method, one of the keys of `TermExtractor.METHODS`
                (e.g. 'tfidf', 'cvalue', 'rake', 'weirdness').
            alpha (float): Weight for \( e_t \) ('basic' and 'combobasic').
            beta (float): Weight for \( e'_t \) ('combobasic').
            threshold (Optional[float]): Minimum score threshold for term inclusion.
            n (int): Maximum length of n-grams to consider (all methods except 'rake').
            num_topics (int): Number of LDA topics ('topicmodeling').
            window_size (int): Context window size ('domaincoherence').
            stoplist (Optional[List[str]]): List of stop words (required by 'rake').
            phrase_delimiters (Optional[List[str]]): List of phrase delimiters (required by 'rake').
            reference_corpus (Optional[Union[Corpus, List[str]]]): General-language reference corpus, required by 'domainpertinence', 'weirdness' and 'relevance'. It must have the same type as the corpus passed to `extract`.
            n_topics (int): Number of NMF topics ('nmf').
            max_features (Optional[int]): Maximum vocabulary size ('nmf').
            stopwords (Union[str, Iterable[str], None]): Drop candidates that start or end with a
                stopword (all methods except 'rake', which uses `stoplist`): "english" (the
                default) for the built-in English list, a list of words, or None to keep every
                candidate. Stopwords inside a multi-word term, as in "degree of freedom", are kept.
            words_layer (Optional[str]): For Teanga input, the span layer holding the words:
                None (the default) for `words` or, if the corpus has no `words` layer,
                `tokens`.
            lemma_layer (Optional[str]): For Teanga input, a sequence layer, based on the
                words layer, holding the lemma of each word (all methods except 'rake').
                If given, words are read as their lemmas, so inflected forms such as
                "cell" and "cells" are counted, and returned, as one term. None (the
                default) reads the words as written.
            pos_pattern (Optional[str]): For Teanga input, keep only the terms whose
                part-of-speech tags match this pattern in at least one occurrence. Tags are
                written in angle brackets, as in `"<ADJ|NOUN>*<NOUN>"`, or use the name of a
                pattern in `termxtract.pos.POS_PATTERNS`, such as `"noun_phrase"`. None (the
                default) keeps every term. Scores are unchanged.
            pos_layer (Optional[str]): The part-of-speech layer used by `pos_pattern`, based
                on the words layer: None (the default) for `upos` or, if the corpus has no
                `upos` layer, `pos`.

        Raises:
            ValueError: If the method is unknown, or a parameter it requires is missing.
        """
        if method not in self.METHODS:
            raise ValueError(
                f"Unknown extraction method: {method!r}. "
                f"Available methods: {', '.join(self.METHODS)}"
            )
        extractor_class, param_names = self.METHODS[method]
        params: dict[str, Any] = {
            "alpha": alpha,
            "beta": beta,
            "threshold": threshold,
            "n": n,
            "num_topics": num_topics,
            "window_size": window_size,
            "stoplist": stoplist,
            "phrase_delimiters": phrase_delimiters,
            "reference_corpus": reference_corpus,
            "n_topics": n_topics,
            "max_features": max_features,
            "stopwords": stopwords,
            "words_layer": words_layer,
            "lemma_layer": lemma_layer,
        }
        self.words_layer = words_layer
        self.lemma_layer = lemma_layer
        self.pos_pattern = (
            None if pos_pattern is None else compile_pos_pattern(pos_pattern)
        )
        self.pos_layer = pos_layer
        self.extractor: BaseTermExtractor = extractor_class(
            **{name: params[name] for name in param_names}
        )

    def extract(
        self,
        corpus: Corpus | list[str],
    ) -> ATEResults:
        """
        Extract terms from a given corpus.

        Args:
            corpus (Union[Corpus, List[str]]): Either a Teanga Corpus object or a list of strings (target corpus).

        Returns:
            ATEResults: An object containing the corpus, terms, and offsets (if applicable).

        Raises:
            ValueError: If the corpus type is not supported or if a required reference corpus is missing.
        """
        if isinstance(corpus, Corpus):
            results = self.extractor.extract_terms_teanga(corpus)
            if self.pos_pattern is None:
                return results
            return filter_by_pos(
                results,
                corpus,
                self.pos_pattern,
                self.pos_layer,
                self.words_layer,
                self.lemma_layer,
            )
        elif isinstance(corpus, list) and all(isinstance(doc, str) for doc in corpus):
            if self.pos_pattern is not None:
                raise ValueError(
                    "pos_pattern needs part-of-speech tags, so it only works with a "
                    "Teanga corpus with a part-of-speech layer."
                )
            return self.extractor.extract_terms_strings(corpus)
        else:
            raise ValueError("The corpus must be a Teanga Corpus or a list of strings.")
