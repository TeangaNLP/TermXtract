import re

from teanga import Corpus

from ..utils import ATEResults
from .base_extractor import BaseTermExtractor


class DomainPertinenceTermExtractor(BaseTermExtractor):
    """Domain Pertinence-based term extraction with n-gram support."""

    def __init__(self, reference_corpus: Corpus | list[str], threshold: float | None = None, n: int = 1):
        """
        Initialize the DomainPertinence extractor.

        Args:
            reference_corpus (Union[Corpus, List[str]]): Reference corpus for computing domain pertinence (Teanga or list of strings).
            threshold (Optional[float]): Minimum score for term inclusion.
            n (int): Maximum n-gram size (e.g., 1 for unigrams, 2 for bigrams, etc.).
        """
        super().__init__(threshold=threshold, n=n)
        self.reference_corpus = reference_corpus

    def extract_terms_teanga(self, corpus: Corpus) -> ATEResults:
        """
        Extract terms from a Teanga corpus using domain pertinence.

        Each n-gram is scored as tf * idf_ref / idf_target, where tf is its relative
        frequency in the document and the IDFs come from the reference and target corpora.

        Args:
            corpus (Corpus): A Teanga corpus with a `words` span layer. The reference corpus
                must also be a Teanga corpus.

        Returns:
            ATEResults: Results with ranked terms and scores, and term offsets.
        """
        # Generate reference IDF from reference corpus
        ref_ngrams = []
        for ref_doc_id in self.reference_corpus.doc_ids:
            ref_doc = self.reference_corpus.doc_by_id(ref_doc_id)
            ref_doc_ngrams = [ngram for ngram, _ in self.generate_ngrams_teanga(self.teanga_words(ref_doc))]
            ref_ngrams.append(ref_doc_ngrams)

        ref_idf = self.compute_idf(ref_ngrams)

        # Generate target corpus n-grams and compute target IDF
        ngrams_by_doc = {}
        for doc_id in corpus.doc_ids:
            doc = corpus.doc_by_id(doc_id)
            ngrams_by_doc[doc_id] = self.generate_ngrams_teanga(self.teanga_words(doc))

        corpus_ngrams = [[ngram for ngram, _ in ngrams_with_offsets] for ngrams_with_offsets in ngrams_by_doc.values()]
        target_idf = self.compute_idf(corpus_ngrams)

        # Compute terms and scores for each document
        terms_by_doc = []
        offsets_by_doc = []
        for doc_id, ngrams_with_offsets in ngrams_by_doc.items():
            tf = self.compute_tf([ngram for ngram, _ in ngrams_with_offsets])
            scores = {
                ngram: tf[ngram] * ref_idf.get(ngram, 0) / max(target_idf.get(ngram, 1), 1e-9)
                for ngram in tf
            }
            doc_terms = self.doc_terms(doc_id, scores)
            terms_by_doc.append(doc_terms)
            offsets_by_doc.append(self.doc_offsets(doc_id, ngrams_with_offsets, doc_terms))

        return ATEResults(corpus=corpus, terms=terms_by_doc, offsets=offsets_by_doc)

    def extract_terms_strings(self, corpus: list[str]) -> ATEResults:
        """
        Extract terms from a list of strings using domain pertinence.

        Each n-gram is scored as tf * idf_ref / idf_target, where tf is its relative
        frequency in the document and the IDFs come from the reference and target corpora.

        Args:
            corpus (List[str]): The documents as plain strings. The reference corpus must
                also be a list of strings.

        Returns:
            ATEResults: Results with ranked terms and scores.
        """
        # Generate reference IDF from reference corpus
        ref_ngrams = [self.generate_ngrams_strings(re.findall(r'\b\w+\b', doc.lower())) for doc in self.reference_corpus]
        ref_idf = self.compute_idf(ref_ngrams)

        tokenized_corpus = [re.findall(r'\b\w+\b', doc.lower()) for doc in corpus]
        corpus_ngrams = [self.generate_ngrams_strings(doc) for doc in tokenized_corpus]
        target_idf = self.compute_idf(corpus_ngrams)

        terms_by_doc = []
        for idx, doc_text in enumerate(corpus):
            tf = self.compute_tf(self.generate_ngrams_strings(tokenized_corpus[idx]))
            scores = {
                ngram: tf[ngram] * ref_idf.get(ngram, 0) / max(target_idf.get(ngram, 1), 1e-9)
                for ngram in tf
            }
            terms_by_doc.append(self.doc_terms(f"doc_{idx}", scores))

        return ATEResults(corpus=corpus, terms=terms_by_doc)
