from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

from teanga import Corpus

from ..pos import compile_pos_pattern, filter_by_pos
from ..stopwords import Stopwords
from ..utils import ATEResults
from .adaboost import DEFAULT_WORDNET_LEXICON, AdaBoostTermExtractor
from .roger import RogerTermExtractor

if TYPE_CHECKING:
    from .tokenextractor import TokenClassificationTermExtractor


def _load_token_classification() -> type["TokenClassificationTermExtractor"]:
    """Import the token-classification extractor, which needs the optional extra."""
    try:
        from .tokenextractor import TokenClassificationTermExtractor
    except ImportError as e:
        raise ImportError(
            "The 'token-classification' method requires the optional 'transformers' "
            "dependencies. Install them with: pip install 'termxtract[transformers]'"
        ) from e
    return TokenClassificationTermExtractor


class SupervisedTermExtractor:
    """
    A wrapper class for selecting the supervised term extraction method.

    `SupervisedTermExtractor.METHODS` maps each method name to a function returning its
    extractor class and the parameters passed to it, so
    `list(SupervisedTermExtractor.METHODS)` lists the available methods.
    """

    METHODS: ClassVar[dict[str, tuple[Callable[[], type], tuple[str, ...]]]] = {
        "adaboost": (
            lambda: AdaBoostTermExtractor,
            (
                "threshold",
                "n",
                "max_depth",
                "n_estimators",
                "estimator",
                "wordnet_lexicon",
                "stopwords",
                "words_layer",
            ),
        ),
        "roger": (
            lambda: RogerTermExtractor,
            (
                "population_size",
                "offspring_size",
                "generations",
                "runs",
                "n",
                "random_state",
                "stopwords",
                "words_layer",
            ),
        ),
        # Loaded on use because torch/transformers are an optional extra
        "token-classification": (
            _load_token_classification,
            (
                "model_name",
                "n",
                "batch_size",
                "epochs",
                "learning_rate",
                "max_length",
                "language",
                "stopwords",
                "words_layer",
            ),
        ),
    }

    extractor: (
        "AdaBoostTermExtractor | RogerTermExtractor | TokenClassificationTermExtractor"
    )

    FILTER_PARAMS: ClassVar[tuple[str, ...]] = ("pos_pattern", "pos_layer")
    """Parameters applied to the results by the wrapper, not passed to the extractor."""

    def __init__(
        self,
        method: str = "adaboost",
        model_name: str = "xlm-roberta-base",
        threshold: float | None = None,
        n: int = 6,
        # Roger parameters
        population_size: int = 20,
        offspring_size: int = 100,
        generations: int = 100,
        runs: int = 21,
        random_state: int | None = None,
        # AdaBoost parameters
        max_depth: int = 1,
        n_estimators: int = 50,
        estimator: str = "decision_tree",
        wordnet_lexicon: str = DEFAULT_WORDNET_LEXICON,
        # Token classification parameters
        batch_size: int = 8,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        max_length: int = 512,
        language: str = "en",
        stopwords: Stopwords = "english",
        words_layer: str | None = None,
        pos_pattern: str | None = None,
        pos_layer: str | None = None,
    ):
        """
        Initialize the TermExtractor with the desired supervised method and parameters.

        Args:
            method (str): The term extraction method, one of the keys of `SupervisedTermExtractor.METHODS`
                ('adaboost', 'roger', 'token-classification').
            model_name (str): Hugging Face model name for token classification.
            threshold (Optional[float]): Minimum score threshold for term inclusion (only for 'adaboost').
            n (int): Maximum length of n-grams to consider.
            population_size (int): Number of parents in each generation (only for 'roger').
            offspring_size (int): Number of offspring in each generation (only for 'roger').
            generations (int): Number of generations in each run (only for 'roger').
            runs (int): Number of independent runs to combine (only for 'roger').
            random_state (Optional[int]): Seed for reproducible scores (only for 'roger').
            max_depth (int): Maximum depth of the decision tree (only for 'adaboost').
            n_estimators (int): Number of boosting iterations (only for 'adaboost').
            estimator (str): Type of base estimator (only for 'adaboost').
            wordnet_lexicon (str): `wn` lexicon for the semantic feature, downloaded on first use if missing (only for 'adaboost').
            batch_size (int): Training batch size (only for 'token-classification').
            epochs (int): Number of training epochs (only for 'token-classification').
            learning_rate (float): Learning rate (only for 'token-classification').
            max_length (int): Maximum sequence length (only for 'token-classification').
            language (str): Language code for tokenization (only for 'token-classification').
            stopwords (Union[str, Iterable[str], None]): Drop extracted terms that start or end
                with a stopword: "english" (the default) for the built-in English list, a list
                of words, or None to keep every term.
            words_layer (Optional[str]): For Teanga input, the span layer holding the words:
                None (the default) for `words` or, if the corpus has no `words` layer,
                `tokens`.
            pos_pattern (Optional[str]): For Teanga input, keep only the terms whose
                part-of-speech tags match this pattern in at least one occurrence. Tags are
                written in angle brackets, as in `"<ADJ|NOUN>*<NOUN>"`, or use the name of a
                pattern in `termxtract.pos.POS_PATTERNS`, such as `"noun_phrase"`. None (the
                default) keeps every term. Scores are unchanged.
            pos_layer (Optional[str]): The part-of-speech layer used by `pos_pattern`, based
                on the words layer: None (the default) for `upos` or, if the corpus has no
                `upos` layer, `pos`.

        Raises:
            ValueError: If the method is unknown.
            ImportError: If the method is 'token-classification' and the optional
                'transformers' dependencies are not installed.
        """
        if method not in self.METHODS:
            raise ValueError(
                f"Unknown supervised extraction method: {method!r}. "
                f"Available methods: {', '.join(self.METHODS)}"
            )
        load_extractor_class, param_names = self.METHODS[method]
        params: dict[str, Any] = {
            "model_name": model_name,
            "threshold": threshold,
            "n": n,
            "population_size": population_size,
            "offspring_size": offspring_size,
            "generations": generations,
            "runs": runs,
            "random_state": random_state,
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "estimator": estimator,
            "wordnet_lexicon": wordnet_lexicon,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "language": language,
            "stopwords": stopwords,
            "words_layer": words_layer,
        }
        self.words_layer = words_layer
        self.pos_pattern = (
            None if pos_pattern is None else compile_pos_pattern(pos_pattern)
        )
        self.pos_layer = pos_layer
        self.extractor = load_extractor_class()(
            **{name: params[name] for name in param_names}
        )

    def extract(self, corpus: Corpus | list[str], labels: dict[str, int]) -> ATEResults:
        """
        Extract terms from a given corpus using supervised learning.

        Args:
            corpus (Union[Corpus, List[str]]): Teanga Corpus object or list of strings.
            labels (Dict[str, int]): Dictionary with terms as keys and binary labels {0, 1}.

        Returns:
            ATEResults: Object containing extracted terms with scores.
        """
        if isinstance(corpus, Corpus):
            results = self.extractor.extract_terms_teanga(corpus, labels)
            if self.pos_pattern is None:
                return results
            return filter_by_pos(
                results, corpus, self.pos_pattern, self.pos_layer, self.words_layer
            )
        elif isinstance(corpus, list) and all(isinstance(doc, str) for doc in corpus):
            if self.pos_pattern is not None:
                raise ValueError(
                    "pos_pattern needs part-of-speech tags, so it only works with a "
                    "Teanga corpus with a part-of-speech layer."
                )
            return self.extractor.extract_terms_strings(corpus, labels)
        else:
            raise ValueError("Corpus must be a Teanga Corpus or a list of strings.")
