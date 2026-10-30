import math

from .reference_extractor import ReferenceCorpusTermExtractor


class RelevanceTermExtractor(ReferenceCorpusTermExtractor):
    """
    Relevance-based term extraction with n-gram support (Peñas et al., 2001).

    Each n-gram is scored as `1 - 1 / log2(2 + p_target(t) * df_target(t) / p_ref(t))`,
    where `p` is the frequency relative to the number of n-grams in each corpus and
    `df_target` is the fraction of target documents containing the n-gram. The reference
    count is add-one smoothed, `p_ref(t) = (count_ref(t) + 1) / N_ref`. Scores are
    between 0 and 1, and are higher for n-grams that are frequent and widespread in the
    target corpus but rare in the reference corpus.
    """

    def score_terms(
        self, target_ngrams: list[list[str]], ref_ngrams: list[list[str]]
    ) -> dict[str, float]:
        """
        Score the n-grams of the target corpus by Relevance.

        Args:
            target_ngrams (List[List[str]]): The n-grams of each target document.
            ref_ngrams (List[List[str]]): The n-grams of each reference document.

        Returns:
            Dict[str, float]: The Relevance of every n-gram in the target corpus, between
            0 and 1.
        """
        target_counts, target_total = self.count_ngrams(target_ngrams)
        ref_counts, ref_total = self.count_ngrams(ref_ngrams)
        doc_counts = self.count_ngrams([list(set(doc)) for doc in target_ngrams])[0]
        num_docs = len(target_ngrams)
        scores = {}
        for ngram, count in target_counts.items():
            p_target = count / target_total
            df_target = doc_counts[ngram] / num_docs
            p_ref = (ref_counts[ngram] + 1) / ref_total
            scores[ngram] = 1 - 1 / math.log2(2 + p_target * df_target / p_ref)
        return scores
