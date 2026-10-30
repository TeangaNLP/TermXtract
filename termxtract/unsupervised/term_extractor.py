from typing import Any, ClassVar

from teanga import Corpus

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
        "tfidf": (TFIDFTermExtractor, ("threshold", "n")),
        "ridf": (RIDFTermExtractor, ("threshold", "n")),
        "cvalue": (CValueTermExtractor, ("threshold", "n")),
        "basic": (BasicTermExtractor, ("alpha", "threshold", "n")),
        "combobasic": (ComboBasicTermExtractor, ("alpha", "beta", "threshold", "n")),
        "domaincoherence": (
            DomainCoherenceTermExtractor,
            ("window_size", "threshold", "n"),
        ),
        "relevance": (RelevanceTermExtractor, ("reference_corpus", "threshold", "n")),
        "topicmodeling": (TopicModelingTermExtractor, ("num_topics", "threshold", "n")),
        "nmf": (NMFTermExtractor, ("threshold", "n_topics", "max_features", "n")),
        "rake": (RAKETermExtractor, ("stoplist", "phrase_delimiters", "threshold")),
        "domainpertinence": (
            DomainPertinenceTermExtractor,
            ("reference_corpus", "threshold", "n"),
        ),
        "weirdness": (WeirdnessTermExtractor, ("reference_corpus", "threshold", "n")),
    }

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
        }
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
            return self.extractor.extract_terms_teanga(corpus)
        elif isinstance(corpus, list) and all(isinstance(doc, str) for doc in corpus):
            return self.extractor.extract_terms_strings(corpus)
        else:
            raise ValueError("The corpus must be a Teanga Corpus or a list of strings.")
