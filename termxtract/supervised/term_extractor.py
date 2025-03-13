from typing import Union, List, Dict, Optional
from teanga import Corpus
from ..utils import ATEResults
from .adaboost import AdaBoostTermExtractor
from .roger import RogerTermExtractor
from .tokenextractor import TokenClassificationTermExtractor


class SupervisedTermExtractor:
    """A wrapper class for selecting the supervised term extraction method."""

    def __init__(
        self,
        method: str = "adaboost",
        model_name: str = "xlm-roberta-base",
        n: int = 6,
        # Roger parameters
        population_size: int = 50,
        generations: int = 100,
        subsample_ratio: float = 0.8,
        # AdaBoost parameters
        max_depth: int = 1,
        n_estimators: int = 50,
        estimator: str = "decision_tree",
        # Token classification parameters
        batch_size: int = 8,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        max_length: int = 512,
        language: str = "en"
    ):
        """
        Initialize the TermExtractor with the desired supervised method and parameters.

        Args:
            method (str): The term extraction method ('adaboost', 'roger', 'token-classification').
            model_name (str): Hugging Face model name for token classification.
            n (int): Maximum length of n-grams to consider.
            
            # Roger parameters
            population_size (int): Number of ranking functions (only for 'roger').
            generations (int): Number of iterations for optimization (only for 'roger').
            subsample_ratio (float): Data fraction used in each iteration (only for 'roger').
            
            # AdaBoost parameters
            max_depth (int): Maximum depth of the decision tree (only for 'adaboost').
            n_estimators (int): Number of boosting iterations (only for 'adaboost').
            estimator (str): Type of base estimator (only for 'adaboost').
            
            # Token classification parameters
            batch_size (int): Training batch size (only for 'token-classification').
            epochs (int): Number of training epochs (only for 'token-classification').
            learning_rate (float): Learning rate (only for 'token-classification').
            max_length (int): Maximum sequence length (only for 'token-classification').
            language (str): Language code for tokenization (only for 'token-classification').
        """
        if method == "adaboost":
            self.extractor = AdaBoostTermExtractor(
                n=n,
                max_depth=max_depth,
                n_estimators=n_estimators,
                estimator=estimator
            )
        elif method == "roger":
            self.extractor = RogerTermExtractor(
                population_size=population_size,
                generations=generations,
                subsample_ratio=subsample_ratio,
                n=n
            )
        elif method == "token-classification":
            self.extractor = TokenClassificationTermExtractor(
                model_name=model_name,
                n=n,
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=learning_rate,
                max_length=max_length,
                language=language
            )
        else:
            raise ValueError(f"Unknown supervised extraction method: {method}")

    def extract(
        self, corpus: Union[Corpus, List[str]], labels: Dict[str, int]
    ) -> ATEResults:
        """
        Extract terms from a given corpus using supervised learning.

        Args:
            corpus (Union[Corpus, List[str]]): Teanga Corpus object or list of strings.
            labels (Dict[str, int]): Dictionary with terms as keys and binary labels {0, 1}.

        Returns:
            ATEResults: Object containing extracted terms with scores.
        """
        if isinstance(corpus, Corpus):
            return self.extractor.extract_terms_teanga(corpus, labels)
        elif isinstance(corpus, list) and all(isinstance(doc, str) for doc in corpus):
            return self.extractor.extract_terms_strings(corpus, labels)
        else:
            raise ValueError("Corpus must be a Teanga Corpus or a list of strings.")
