import math
import re
from collections import Counter

from teanga import Corpus

from ..utils import ATEResults
from .base_extractor import BaseTermExtractor


class RIDFTermExtractor(BaseTermExtractor):
    r"""
    Residual IDF term extraction (Church & Gale, 1999), as in ATR4S (Astrakhantsev,
    2018) and JATE 2.0.

    Residual IDF is the amount by which a candidate's IDF exceeds the IDF that a Poisson
    distribution of its occurrences over the documents would predict:
    \( \log_2 \frac{D}{df(t)} + \log_2 (1 - e^{-cf(t) / D}) \), where \( D \) is the
    number of documents, \( df(t) \) the number of documents containing \( t \) and
    \( cf(t) \) its number of occurrences. Terms concentrated in a few documents score
    highest. Scores are computed over the whole corpus, so a term has the same score in
    every document it occurs in.
    """

    def compute_ridf(self, corpus_ngrams: list[list[str]]) -> dict[str, float]:
        """
        Compute Residual Inverse Document Frequency (RIDF) for n-grams.

        Args:
            corpus_ngrams (List[List[str]]): List of n-grams for each document.

        Returns:
            Dict[str, float]: RIDF values for each n-gram.
        """
        num_docs = len(corpus_ngrams)
        collection_freq = Counter(ngram for doc in corpus_ngrams for ngram in doc)
        doc_freq = Counter(ngram for doc in corpus_ngrams for ngram in set(doc))
        return {
            ngram: math.log2(num_docs / doc_freq[ngram])
            # log2(1 - e^-x), with expm1 for precision when x is small
            + math.log2(-math.expm1(-cf / num_docs))
            for ngram, cf in collection_freq.items()
        }

    def extract_terms_teanga(self, corpus: Corpus) -> ATEResults:
        """
        Extract terms from a Teanga corpus using RIDF.

        Args:
            corpus: A Teanga Corpus object.

        Returns:
            ATEResults: Results with ranked terms and scores, and term offsets.
        """
        ngrams_by_doc = {
            doc_id: self.generate_ngrams_teanga(
                self.teanga_words(corpus.doc_by_id(doc_id))
            )
            for doc_id in corpus.doc_ids
        }
        ridf_scores = self.compute_ridf(
            [[ngram for ngram, _ in ngrams] for ngrams in ngrams_by_doc.values()]
        )

        terms_by_doc = []
        offsets_by_doc = []
        for doc_id, ngrams_with_offsets in ngrams_by_doc.items():
            doc_terms = self.doc_terms(
                doc_id, {ngram: ridf_scores[ngram] for ngram, _ in ngrams_with_offsets}
            )
            terms_by_doc.append(doc_terms)
            offsets_by_doc.append(
                self.doc_offsets(doc_id, ngrams_with_offsets, doc_terms)
            )

        return ATEResults(corpus=corpus, terms=terms_by_doc, offsets=offsets_by_doc)

    def extract_terms_strings(self, corpus: list[str]) -> ATEResults:
        """
        Extract terms from a plain list of strings using RIDF.

        Args:
            corpus (List[str]): List of documents as strings.

        Returns:
            ATEResults: Results with ranked terms and scores.
        """
        corpus_ngrams = [
            self.generate_ngrams_strings(re.findall(r"\b\w+\b", doc.lower()))
            for doc in corpus
        ]
        ridf_scores = self.compute_ridf(corpus_ngrams)

        terms_by_doc = [
            self.doc_terms(
                f"doc_{idx}", {ngram: ridf_scores[ngram] for ngram in ngrams}
            )
            for idx, ngrams in enumerate(corpus_ngrams)
        ]
        processed_corpus = [
            {"doc_id": f"doc_{idx}", "text": doc} for idx, doc in enumerate(corpus)
        ]
        return ATEResults(corpus=processed_corpus, terms=terms_by_doc)
