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
from .reference_extractor import ReferenceCorpus
from .relevance import RelevanceTermExtractor
from .ridf import RIDFTermExtractor
from .tfidf import TFIDFTermExtractor
from .weirdness import WeirdnessTermExtractor

CANDIDATE_PARAMS = (
    "threshold",
    "n",
    "stopwords",
    "words_layer",
    "lemma_layer",
    "min_frequency",
    "word_pattern",
    "pos_pattern",
    "pos_layer",
)
"""The parameters of `BaseTermExtractor`, which every method except RAKE takes."""


class TermExtractor:
    """
    A wrapper class for selecting the term extraction method.

    `TermExtractor.METHODS` maps each method name to its extractor class and the
    parameters passed to it, so `list(TermExtractor.METHODS)` lists the available methods.
    """

    METHODS: ClassVar[dict[str, tuple[type[BaseTermExtractor], tuple[str, ...]]]] = {
        "tfidf": (TFIDFTermExtractor, CANDIDATE_PARAMS),
        "ridf": (RIDFTermExtractor, CANDIDATE_PARAMS),
        "cvalue": (CValueTermExtractor, CANDIDATE_PARAMS),
        "basic": (BasicTermExtractor, ("alpha", *CANDIDATE_PARAMS)),
        "combobasic": (ComboBasicTermExtractor, ("alpha", "beta", *CANDIDATE_PARAMS)),
        "domaincoherence": (
            DomainCoherenceTermExtractor,
            ("window_size", "seed_terms", "context_words", *CANDIDATE_PARAMS),
        ),
        "relevance": (RelevanceTermExtractor, ("reference_corpus", *CANDIDATE_PARAMS)),
        "topicmodeling": (
            TopicModelingTermExtractor,
            ("num_topics", "top_words", *CANDIDATE_PARAMS),
        ),
        "nmf": (NMFTermExtractor, ("n_topics", "max_features", *CANDIDATE_PARAMS)),
        "rake": (RAKETermExtractor, ("stoplist", "phrase_delimiters", "threshold")),
        "domainpertinence": (
            DomainPertinenceTermExtractor,
            ("reference_corpus", "tie_break", *CANDIDATE_PARAMS),
        ),
        "weirdness": (WeirdnessTermExtractor, ("reference_corpus", *CANDIDATE_PARAMS)),
    }

    def __init__(
        self,
        method: str = "tfidf",
        alpha: float = 0.5,
        beta: float = 0.5,
        threshold: float | None = None,
        n: int = 1,
        num_topics: int = 20,
        top_words: int = 200,
        window_size: int = 5,
        seed_terms: int = 200,
        context_words: int = 50,
        stoplist: list[str] | None = None,
        phrase_delimiters: list[str] | None = None,
        reference_corpus: ReferenceCorpus | list[ReferenceCorpus] | None = None,
        tie_break: str | None = "ratio",
        n_topics: int = 10,
        max_features: int | None = None,
        stopwords: Stopwords = "english",
        words_layer: str | None = None,
        lemma_layer: str | None = None,
        min_frequency: int = 1,
        word_pattern: str | None = "alphanumeric",
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
            top_words (int): Number of most probable words of each topic that count
                towards the scores ('topicmodeling').
            window_size (int): The number of words on either side of a candidate that
                count as its context ('domaincoherence').
            seed_terms (int): The number of best candidates by Basic used to choose the
                context words ('domaincoherence').
            context_words (int): The number of context words that candidates are scored
                against ('domaincoherence').
            stoplist (Optional[List[str]]): List of stop words (required by 'rake').
            phrase_delimiters (Optional[List[str]]): List of phrase delimiters (required by 'rake').
            reference_corpus (Optional[Union[Corpus, List[str], List[Union[Corpus, List[str]]]]]):
                General-language reference corpus, required by 'domainpertinence',
                'weirdness' and 'relevance'. It must have the same type as the corpus
                passed to `extract`. A list of reference corpora is compared one by one
                by 'domainpertinence', and read as one corpus by the others.
            tie_break (Optional[str]): How to order the candidates that tie at 1
                ('domainpertinence'): "ratio", "frequency" or None.
            n_topics (int): Number of NMF topics ('nmf').
            max_features (Optional[int]): Maximum vocabulary size ('nmf').
            stopwords (Union[str, Iterable[str], None]): Drop candidates that start or end with a
                stopword (all methods except 'rake', which uses `stoplist`): the name of a
                built-in list in `termxtract.stopwords.STOPWORD_LISTS` ("english", the
                default, "hindi" or "slovenian"), a list of words, or None to keep every
                candidate. Stopwords inside a multi-word term, as in "degree of freedom", are kept.
            words_layer (Optional[str]): For Teanga input, the span layer holding the words:
                None (the default) for `words` or, if the corpus has no `words` layer,
                `tokens`.
            lemma_layer (Optional[str]): For Teanga input, a sequence layer, based on the
                words layer, holding the lemma of each word (all methods except 'rake').
                If given, words are read as their lemmas, so inflected forms such as
                "cell" and "cells" are counted, and returned, as one term. None (the
                default) reads the words as written.
            min_frequency (int): Drop candidates that occur fewer times than this in the
                corpus (all methods except 'rake'). ATR4S uses 2. The default of 1 keeps
                every candidate.
            word_pattern (Optional[str]): Drop candidates with a word that does not fully
                match this regular expression (all methods except 'rake'), or the name of
                a pattern in `termxtract.candidates.WORD_PATTERNS`: "alphanumeric" (the
                default) drops words without a letter or digit, such as "%", and
                "atr4s" keeps words of three or more letters, digits and hyphens, as
                ATR4S's noise filter does. None keeps every word.
            pos_pattern (Optional[str]): For Teanga input, count only the occurrences of
                candidates whose part-of-speech tags match this pattern, so a term must
                match it in at least one occurrence. Tags are written in angle brackets,
                as in `"<ADJ|NOUN>*<NOUN>"`, or use the name of a pattern in
                `termxtract.pos.POS_PATTERNS`, such as `"noun_phrase"`. None (the
                default) allows every tag sequence. RAKE's terms are filtered after
                scoring instead, so its scores are unchanged.
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
            "top_words": top_words,
            "window_size": window_size,
            "seed_terms": seed_terms,
            "context_words": context_words,
            "stoplist": stoplist,
            "phrase_delimiters": phrase_delimiters,
            "reference_corpus": reference_corpus,
            "tie_break": tie_break,
            "n_topics": n_topics,
            "max_features": max_features,
            "stopwords": stopwords,
            "words_layer": words_layer,
            "lemma_layer": lemma_layer,
            "min_frequency": min_frequency,
            "word_pattern": word_pattern,
            "pos_pattern": pos_pattern,
            "pos_layer": pos_layer,
        }
        self.words_layer = words_layer
        self.lemma_layer = lemma_layer
        self.pos_pattern = (
            None if pos_pattern is None else compile_pos_pattern(pos_pattern)
        )
        self.pos_layer = pos_layer
        # Methods that take pos_pattern count only the matching occurrences; the terms
        # of the others are filtered after extraction
        self.filters_by_pos = "pos_pattern" not in param_names
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
            if self.pos_pattern is None or not self.filters_by_pos:
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
