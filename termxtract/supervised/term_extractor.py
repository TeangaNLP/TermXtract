from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from teanga import Corpus

from ..datasets.base import GOLD_LAYER
from ..pos import compile_pos_pattern, filter_by_pos
from ..stopwords import Stopwords
from ..utils import ATEResults
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
        "roger": (
            lambda: RogerTermExtractor,
            (
                "population_size",
                "offspring_size",
                "generations",
                "runs",
                "n",
                "random_state",
                "max_negatives",
                "patience",
                "n_jobs",
                "stopwords",
                "words_layer",
                "terms_layer",
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
                "random_state",
                "patience",
                "terms_layer",
                "sentences_layer",
            ),
        ),
    }

    extractor: "RogerTermExtractor | TokenClassificationTermExtractor"

    FILTER_PARAMS: ClassVar[tuple[str, ...]] = ("pos_pattern", "pos_layer")
    """Parameters applied to the results by the wrapper, not passed to the extractor."""

    def __init__(
        self,
        method: str,
        model_name: str = "xlm-roberta-base",
        n: int | None = None,
        # Roger parameters
        population_size: int = 20,
        offspring_size: int = 100,
        generations: int = 100,
        runs: int = 21,
        random_state: int | None = None,
        max_negatives: int | None = 10_000,
        patience: int | None = None,
        n_jobs: int | None = None,
        # Token classification parameters
        batch_size: int = 8,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        max_length: int = 512,
        language: str = "en",
        stopwords: Stopwords = "english",
        words_layer: str | None = None,
        terms_layer: str = GOLD_LAYER,
        sentences_layer: str | None = None,
        pos_pattern: str | None = None,
        pos_layer: str | None = None,
    ):
        """
        Initialize the TermExtractor with the desired supervised method and parameters.

        Args:
            method (str): The term extraction method, one of the keys of `SupervisedTermExtractor.METHODS`
                ('roger', 'token-classification').
            model_name (str): Hugging Face model name for token classification, or the
                directory of a model saved with `save`.
            n (Optional[int]): Maximum length of n-grams to consider, or None for the
                method's default: 2 for 'roger' and 6 for 'token-classification'.
            population_size (int): Number of parents in each generation (only for 'roger').
            offspring_size (int): Number of offspring in each generation (only for 'roger').
            generations (int): Number of generations in each run (only for 'roger').
            runs (int): Number of independent runs to combine (only for 'roger').
            random_state (Optional[int]): Seed for reproducible scores, or for
                'token-classification' for reproducible training.
            max_negatives (Optional[int]): Train on at most this many candidates
                labelled 0, sampled for each run (only for 'roger').
            patience (Optional[int]): For 'roger', stop a run when it has not improved
                for this many generations; for 'token-classification', stop training
                when the F1 on the validation set has not improved for this many
                epochs.
            n_jobs (Optional[int]): Number of runs trained in parallel, -1 for one per
                CPU (only for 'roger').
            batch_size (int): Training batch size (only for 'token-classification').
            epochs (int): Number of training epochs (only for 'token-classification').
            learning_rate (float): Learning rate (only for 'token-classification').
            max_length (int): Maximum sequence length (only for 'token-classification').
            language (str): Language code for tokenization (only for 'token-classification').
            stopwords (Union[str, Iterable[str], None]): Drop extracted terms that start or end
                with a stopword: "english" (the default) or "hindi" for a built-in list, a
                list of words, or None to keep every term.
            words_layer (Optional[str]): For Teanga input, the span layer holding the words:
                None (the default) for `words` or, if the corpus has no `words` layer,
                `tokens`.
            terms_layer (str): For Teanga input, the span layer of gold terms that
                `fit` trains on when it is given no labels: `terms` by default, as in
                the corpora of `termxtract.datasets`.
            sentences_layer (Optional[str]): For Teanga input, the span layer of the
                sentences: None (the default) for `sentences` or, if the corpus has no
                such layer, splitting after each word that is ".", "!" or "?" (only for
                'token-classification').
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
            "n": n,
            "population_size": population_size,
            "offspring_size": offspring_size,
            "generations": generations,
            "runs": runs,
            "random_state": random_state,
            "max_negatives": max_negatives,
            "patience": patience,
            "n_jobs": n_jobs,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "language": language,
            "stopwords": stopwords,
            "words_layer": words_layer,
            "terms_layer": terms_layer,
            "sentences_layer": sentences_layer,
        }
        self.method = method
        self.words_layer = words_layer
        self.pos_pattern = (
            None if pos_pattern is None else compile_pos_pattern(pos_pattern)
        )
        self.pos_layer = pos_layer
        # Leave out n when it is None, so the extractor uses its own default
        self.extractor = load_extractor_class()(
            **{
                name: params[name]
                for name in param_names
                if name != "n" or n is not None
            }
        )

    def fit(
        self,
        corpus: Corpus | list[str],
        labels: dict[str, int] | None = None,
        validation: Corpus | list[str] | None = None,
    ) -> "SupervisedTermExtractor":
        """
        Train the method on a corpus.

        Args:
            corpus (Union[Corpus, List[str]]): Teanga Corpus object or list of strings.
            labels (Optional[Dict[str, int]]): Dictionary with terms as keys and binary
                labels {0, 1}. Roger needs examples of both; given only terms, it counts
                every other candidate of the corpus as a non-term. None, for a Teanga
                corpus, trains on the gold terms of `terms_layer`: token classification
                tags each occurrence of them, and Roger takes them as the terms.
            validation (Optional[Union[Corpus, List[str]]]): For 'token-classification',
                a corpus of the same kind, labelled in the same way, to compute the F1
                of the tagged terms on after each epoch. The model of the best epoch is
                kept, and `patience` can stop training early.

        Returns:
            SupervisedTermExtractor: This extractor, trained, so that `fit(...).predict(...)`
            can be chained.

        Raises:
            ValueError: If the corpus is neither a Teanga Corpus nor a list of strings,
                `pos_pattern` is set and the corpus is a list of strings, a list of
                strings is given no labels, or `validation` is given for Roger.
        """
        teanga = self._is_teanga(corpus)
        if labels is None and not teanga:
            raise ValueError(
                "A list of strings has no gold terms, so fit needs labels for it."
            )
        if validation is None:
            options = {}
        elif self.method == "token-classification":
            options = {"validation": validation}
        else:
            raise ValueError(f"The {self.method!r} method takes no validation set.")
        if teanga:
            self.extractor.fit_teanga(corpus, labels, **options)
        else:
            assert labels is not None
            self.extractor.fit_strings(corpus, labels, **options)
        return self

    def predict(self, corpus: Corpus | list[str]) -> ATEResults:
        """
        Extract terms from a corpus with the trained method.

        The corpus can differ from the one the method was trained on, for example
        another domain or another split of a dataset.

        Args:
            corpus (Union[Corpus, List[str]]): Teanga Corpus object or list of strings.

        Returns:
            ATEResults: The terms of each document with their scores, and their offsets
            for a Teanga corpus.

        Raises:
            ValueError: If the corpus is neither a Teanga Corpus nor a list of strings,
                or `pos_pattern` is set and the corpus is a list of strings.
        """
        if not self._is_teanga(corpus):
            return self.extractor.predict_strings(corpus)
        results = self.extractor.predict_teanga(corpus)
        if self.pos_pattern is None:
            return results
        return filter_by_pos(
            results, corpus, self.pos_pattern, self.pos_layer, self.words_layer
        )

    def extract(
        self, corpus: Corpus | list[str], labels: dict[str, int] | None = None
    ) -> ATEResults:
        """
        Train the method on a corpus and extract the terms of that same corpus.

        To extract terms from another corpus, use `fit` and then `predict`.

        Args:
            corpus (Union[Corpus, List[str]]): Teanga Corpus object or list of strings.
            labels (Optional[Dict[str, int]]): Dictionary with terms as keys and binary
                labels {0, 1}, or None to train on the gold terms of `terms_layer`, as
                for `fit`.

        Returns:
            ATEResults: Object containing extracted terms with scores.
        """
        return self.fit(corpus, labels).predict(corpus)

    def save(self, directory: str | Path) -> None:
        """
        Save a trained token-classification model, to pass the directory as
        `model_name` to extract terms without training again.

        Args:
            directory (Union[str, Path]): The directory to save to.

        Raises:
            ValueError: If the method is not 'token-classification'.
            RuntimeError: If the model has not been trained.
        """
        if self.method != "token-classification":
            raise ValueError(f"The {self.method!r} method cannot be saved.")
        self.extractor.save(directory)  # type: ignore[union-attr]

    def _is_teanga(self, corpus: Corpus | list[str]) -> bool:
        """
        Check the type of a corpus.

        Returns:
            bool: True for a Teanga Corpus, False for a list of strings.

        Raises:
            ValueError: If the corpus is neither, or `pos_pattern` is set and the corpus
                is a list of strings.
        """
        if isinstance(corpus, Corpus):
            return True
        if not (
            isinstance(corpus, list) and all(isinstance(doc, str) for doc in corpus)
        ):
            raise ValueError("Corpus must be a Teanga Corpus or a list of strings.")
        if self.pos_pattern is not None:
            raise ValueError(
                "pos_pattern needs part-of-speech tags, so it only works with a "
                "Teanga corpus with a part-of-speech layer."
            )
        return False
