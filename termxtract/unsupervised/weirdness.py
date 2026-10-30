from .reference_extractor import ReferenceCorpusTermExtractor


class WeirdnessTermExtractor(ReferenceCorpusTermExtractor):
    """
    Weirdness-based term extraction with n-gram support (Ahmad et al., 1999).

    Each n-gram is scored as `p_target(t) / p_ref(t)`: how many times more frequent it
    is in the target corpus than in the reference corpus. Frequencies are relative to
    the number of n-grams in each corpus, and the reference count is add-one smoothed,
    `p_ref(t) = (count_ref(t) + 1) / N_ref`, so n-grams missing from the reference
    corpus get a finite score.
    """

    def score_terms(
        self, target_ngrams: list[list[str]], ref_ngrams: list[list[str]]
    ) -> dict[str, float]:
        """
        Score the n-grams of the target corpus by Weirdness.

        Args:
            target_ngrams (List[List[str]]): The n-grams of each target document.
            ref_ngrams (List[List[str]]): The n-grams of each reference document.

        Returns:
            Dict[str, float]: The Weirdness of every n-gram in the target corpus.
        """
        target_counts, target_total = self.count_ngrams(target_ngrams)
        ref_counts, ref_total = self.count_ngrams(ref_ngrams)
        return {
            ngram: (count / target_total) / ((ref_counts[ngram] + 1) / ref_total)
            for ngram, count in target_counts.items()
        }
