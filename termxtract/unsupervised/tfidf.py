import re
from collections import Counter
import math
from typing import List, Dict, Optional, Tuple
from .utils import ATEResults


class TFIDFTermExtractor:
    """TF-IDF based term extraction with n-gram support and a threshold option."""

    def __init__(self, threshold: Optional[float] = None, n: int = 1):
        self.threshold = threshold
        self.n = n

    def generate_ngrams_teanga(self, words_with_offsets: List[Tuple[int, int, str]]) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Generate n-grams with offsets for a Teanga corpus.

        Args:
            words_with_offsets (List[Tuple[int, int, str]]): List of (start, end, text) tuples for words.

        Returns:
            List[Tuple[str, Tuple[int, int]]]: List of n-grams with their start and end offsets.
        """
        ngrams = []
        words = [text for _, _, text in words_with_offsets]
        for i in range(len(words)):
            for j in range(1, self.n + 1):
                if i + j <= len(words):
                    ngram = " ".join(words[i:i + j])
                    start_offset = words_with_offsets[i][0]  # Start offset of the first word in the n-gram
                    end_offset = words_with_offsets[i + j - 1][1]  # End offset of the last word in the n-gram
                    ngrams.append((ngram, (start_offset, end_offset)))
        return ngrams

    def generate_ngrams_strings(self, words: List[str]) -> List[str]:
        """
        Generate n-grams for a plain list of strings.

        Args:
            words (List[str]): List of words in the document.

        Returns:
            List[str]: List of n-grams.
        """
        ngrams = []
        for i in range(len(words)):
            for j in range(1, self.n + 1):
                if i + j <= len(words):
                    ngram = " ".join(words[i:i + j])
                    ngrams.append(ngram)
        return ngrams

    def compute_tf(self, ngrams: List[str]) -> Dict[str, float]:
        ngram_count = len(ngrams)
        term_frequencies = Counter(ngrams)
        return {ngram: count / ngram_count for ngram, count in term_frequencies.items()}

    def compute_idf(self, corpus_ngrams: List[List[str]]) -> Dict[str, float]:
        num_docs = len(corpus_ngrams)
        idf = {}
        all_ngrams = set(ngram for doc in corpus_ngrams for ngram in doc)
        for ngram in all_ngrams:
            doc_count = sum(1 for doc in corpus_ngrams if ngram in doc)
            idf[ngram] = math.log(num_docs / (1 + doc_count))
        return idf

    def extract_terms_teanga(self, corpus) -> ATEResults:
        corpus.add_layer_meta("terms", layer_type="span", base="text")
        ngrams_by_doc = {}
        for doc_id in corpus.doc_ids:
            doc = corpus.doc_by_id(doc_id)
            words_with_offsets = [(start, end, doc.text[start:end]) for start, end in doc.words]
            ngrams_with_offsets = self.generate_ngrams_teanga(words_with_offsets)
            ngrams_by_doc[doc_id] = ngrams_with_offsets

        corpus_ngrams = [[ngram for ngram, _ in ngrams_with_offsets] for ngrams_with_offsets in ngrams_by_doc.values()]
        idf = self.compute_idf(corpus_ngrams)

        terms_offsets_by_doc = []
        terms_by_doc = []
        for doc_id, ngrams_with_offsets in ngrams_by_doc.items():
            ngrams = [ngram for ngram, _ in ngrams_with_offsets]
            tf = self.compute_tf(ngrams)
            tfidf = {ngram: tf[ngram] * idf[ngram] for ngram in tf}

            if self.threshold is not None:
                tfidf = {ngram: score for ngram, score in tfidf.items() if score >= self.threshold}

            terms = [ngram for ngram in tfidf.keys()]
            terms_by_doc.append({"doc_id": doc_id, "terms": terms})

            offsets = [{"term": ngram, "offset": offset} for ngram, offset in ngrams_with_offsets if ngram in tfidf]
            terms_offsets_by_doc.append({"doc_id": doc_id, "offsets": offsets})

        return ATEResults(corpus=corpus, terms=terms_by_doc, offsets=terms_offsets_by_doc)

    def extract_terms_strings(self, corpus: List[str]) -> ATEResults:
        tokenized_corpus = [re.findall(r'\b\w+\b', doc.lower()) for doc in corpus]
        corpus_ngrams = [self.generate_ngrams_strings(doc) for doc in tokenized_corpus]
        idf = self.compute_idf(corpus_ngrams)

        terms_by_doc = []
        processed_corpus = []
        for idx, doc_text in enumerate(corpus):
            doc_id = f"doc_{idx}"
            processed_corpus.append({"doc_id": doc_id, "text": doc_text})

            doc_ngrams = self.generate_ngrams_strings(tokenized_corpus[idx])
            tf = self.compute_tf(doc_ngrams)
            tfidf = {ngram: tf[ngram] * idf[ngram] for ngram in tf}

            if self.threshold is not None:
                tfidf = {ngram: score for ngram, score in tfidf.items() if score >= self.threshold}

            terms = [ngram for ngram in tfidf.keys()]
            terms_by_doc.append({"doc_id": doc_id, "terms": terms})

        return ATEResults(corpus=processed_corpus, terms=terms_by_doc)
