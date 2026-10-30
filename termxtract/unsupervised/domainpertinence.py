from .reference_extractor import ReferenceCorpusTermExtractor


class DomainPertinenceTermExtractor(ReferenceCorpusTermExtractor):
    """
    Domain Pertinence-based term extraction with n-gram support (Navigli & Velardi, 2004).

    Each n-gram is scored as `p_target(t) / max(p_target(t), p_ref(t))`, where the
    frequencies are relative to the number of n-grams in each corpus. The score is 1 for
    n-grams that are at least as frequent in the target corpus as in the reference
    corpus, including those missing from the reference corpus, and lower for n-grams that
    are more typical of the reference corpus. Domain Pertinence is best used with a
    `threshold` to filter candidates, as many terms tie at 1.
    """

    def score_terms(
        self, target_ngrams: list[list[str]], ref_ngrams: list[list[str]]
    ) -> dict[str, float]:
        """
        Score the n-grams of the target corpus by Domain Pertinence.

        Args:
            target_ngrams (List[List[str]]): The n-grams of each target document.
            ref_ngrams (List[List[str]]): The n-grams of each reference document.

        Returns:
            Dict[str, float]: The Domain Pertinence of every n-gram in the target corpus,
            between 0 and 1.
        """
        target_counts, target_total = self.count_ngrams(target_ngrams)
        ref_counts, ref_total = self.count_ngrams(ref_ngrams)
        scores = {}
        for ngram, count in target_counts.items():
            p_target = count / target_total
            p_ref = ref_counts[ngram] / ref_total
            scores[ngram] = p_target / max(p_target, p_ref)
        return scores
