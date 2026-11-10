from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from ..candidates import split_words
from ..stopwords import Stopwords
from ..utils import ATEResults
from .base_extractor import BaseTermExtractor
from .ldaextractor import _words


class NMFTermExtractor(BaseTermExtractor):
    """NMF-based term extraction with n-gram support."""

    def __init__(
        self,
        threshold: float | None = None,
        n: int = 1,
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
        """
        Initialize the NMF extractor.

        Args:
            threshold (Optional[float]): Minimum score threshold for term inclusion.
            n (int): Maximum n-gram size (e.g., 1 for unigrams, 2 for bigrams, etc.).
            n_topics (int): Number of topics for NMF-based extraction.
            max_features (Optional[int]): Maximum number of features for TF-IDF vectorization.
            stopwords (Union[str, Iterable[str], None]): Drop candidates that start or end
                with a stopword: "english", "hindi", a list of words, or None.
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
            pos_layer (Optional[str]): The part-of-speech layer for `pos_pattern`, or
                None for `upos` or, failing that, `pos`.
        """
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
        self.n_topics = n_topics
        self.max_features = max_features

    def score_terms(self, docs: list[list[str]]) -> dict[str, float]:
        """
        Score the candidates of a corpus by their largest weight in any NMF topic.

        Args:
            docs (List[List[str]]): The occurrences of candidates in each document.

        Returns:
            Dict[str, float]: The score of each candidate.
        """
        # Pass the candidates through as they are, so both input types share them
        vectorizer = TfidfVectorizer(
            max_features=self.max_features, analyzer=_words, lowercase=False
        )
        tfidf_matrix = vectorizer.fit_transform(docs)
        feature_names = vectorizer.get_feature_names_out()

        nmf = NMF(n_components=self.n_topics, random_state=42)
        nmf.fit(tfidf_matrix)
        H = nmf.components_
        return {feature_names[i]: max(H[:, i]) for i in range(len(feature_names))}

    def extract_terms_strings(self, corpus: list[str]) -> ATEResults:
        """Extract terms from a plain list of strings using NMF."""
        corpus_ngrams = self.candidate_ngrams_strings(
            [split_words(doc) for doc in corpus]
        )
        term_scores = self.score_terms(corpus_ngrams)
        # A term has the same score in every document, but is listed only in those it
        # occurs in. With max_features, some candidates have no score.
        terms_by_doc = [
            self.doc_terms(
                f"doc_{idx}",
                {ngram: term_scores[ngram] for ngram in ngrams if ngram in term_scores},
            )
            for idx, ngrams in enumerate(corpus_ngrams)
        ]
        return ATEResults(corpus=corpus, terms=terms_by_doc)

    def extract_terms_teanga(self, corpus) -> ATEResults:
        """Extract terms from a Teanga corpus using NMF."""
        ngrams_by_doc = self.candidate_ngrams_teanga(corpus)
        term_scores = self.score_terms(
            [[ngram for ngram, _ in ngrams] for ngrams in ngrams_by_doc.values()]
        )
        terms_by_doc = []
        offsets_by_doc = []
        for doc_id, ngrams_with_offsets in ngrams_by_doc.items():
            doc_terms = self.doc_terms(
                doc_id,
                {
                    ngram: term_scores[ngram]
                    for ngram, _ in ngrams_with_offsets
                    if ngram in term_scores
                },
            )
            terms_by_doc.append(doc_terms)
            offsets_by_doc.append(
                self.doc_offsets(doc_id, ngrams_with_offsets, doc_terms)
            )
        return ATEResults(corpus=corpus, terms=terms_by_doc, offsets=offsets_by_doc)
