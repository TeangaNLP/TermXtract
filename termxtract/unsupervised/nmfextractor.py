import re

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from ..stopwords import Stopwords
from ..utils import ATEResults
from .base_extractor import BaseTermExtractor


class NMFTermExtractor(BaseTermExtractor):
    """NMF-based term extraction with n-gram support."""

    def __init__(
        self,
        threshold: float | None = None,
        n: int = 1,
        n_topics: int = 10,
        max_features: int | None = None,
        stopwords: Stopwords = "english",
    ):
        """
        Initialize the NMF extractor.

        Args:
            threshold (Optional[float]): Minimum score threshold for term inclusion.
            n (int): Maximum n-gram size (e.g., 1 for unigrams, 2 for bigrams, etc.).
            n_topics (int): Number of topics for NMF-based extraction.
            max_features (Optional[int]): Maximum number of features for TF-IDF vectorization.
            stopwords (Union[str, Iterable[str], None]): Drop candidates that start or end
                with a stopword: "english", a list of words, or None.
        """
        super().__init__(threshold=threshold, n=n, stopwords=stopwords)
        self.n_topics = n_topics
        self.max_features = max_features

    def score_terms(self, docs: list[list[str]]) -> dict[str, float]:
        """
        Score the n-grams of a corpus by their largest weight in any NMF topic.

        Args:
            docs (List[List[str]]): The words of each document.

        Returns:
            Dict[str, float]: The score of each n-gram.
        """
        # Pass the words through as they are, so both input types are tokenized once
        vectorizer = TfidfVectorizer(
            max_features=self.max_features, analyzer=self.generate_ngrams_strings
        )
        tfidf_matrix = vectorizer.fit_transform(docs)
        feature_names = vectorizer.get_feature_names_out()

        nmf = NMF(n_components=self.n_topics, random_state=42)
        nmf.fit(tfidf_matrix)
        H = nmf.components_
        return {feature_names[i]: max(H[:, i]) for i in range(len(feature_names))}

    def extract_terms_strings(self, corpus: list[str]) -> ATEResults:
        """Extract terms from a plain list of strings using NMF."""
        term_scores = self.score_terms(
            [re.findall(r"\b\w+\b", doc.lower()) for doc in corpus]
        )
        terms_by_doc = [
            self.doc_terms(f"doc_{idx}", term_scores) for idx in range(len(corpus))
        ]
        return ATEResults(corpus=corpus, terms=terms_by_doc)

    def extract_terms_teanga(self, corpus) -> ATEResults:
        """Extract terms from a Teanga corpus using NMF."""
        term_scores = self.score_terms(
            [
                [text for _, _, text in self.teanga_words(corpus.doc_by_id(doc_id))]
                for doc_id in corpus.doc_ids
            ]
        )
        terms_by_doc = [
            self.doc_terms(doc_id, term_scores) for doc_id in corpus.doc_ids
        ]
        return ATEResults(corpus=corpus, terms=terms_by_doc)
