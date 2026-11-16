import unittest

from teanga import Corpus

from termxtract.unsupervised.rake import RAKETermExtractor


class TestRAKE(unittest.TestCase):
    """Test cases for RAKE-based term extraction."""

    def setUp(self):
        """Set up common variables for testing."""
        self.stoplist = ["is", "the", "and", "of", "a", "an", "to", "for"]
        self.delimiters = [".", ",", "!", "?", "-", ";"]
        self.threshold = 1.0  # Default threshold for the tests
        self.extractor = RAKETermExtractor(
            stoplist=self.stoplist,
            phrase_delimiters=self.delimiters,
            threshold=self.threshold,
        )

    def test_rake_extraction_teanga(self):
        """Test RAKE extraction on a Teanga corpus."""
        # Step 1: Create a Teanga corpus
        corpus = Corpus()
        corpus.add_layer_meta("text")
        corpus.add_layer_meta("words", layer_type="span", base="text")

        # Add documents
        doc1 = corpus.add_doc("RAKE identifies key phrases from text efficiently.")
        doc1.words = [(0, 4), (5, 14), (15, 18), (19, 25), (26, 30), (31, 41)]

        doc2 = corpus.add_doc(
            "Rapid Automatic Keyword Extraction is simple and effective."
        )
        doc2.words = [(0, 5), (6, 15), (16, 23), (24, 32), (33, 35), (36, 42), (43, 46)]

        # Extract terms using RAKE
        results = self.extractor.extract_terms_teanga(corpus)

        # Assert results
        self.assertIsNotNone(results.terms, "Terms should not be None.")
        self.assertTrue(
            all("doc_id" in item for item in results.terms),
            "Each result should have 'doc_id'.",
        )

        for term_entry in results.terms:
            self.assertIn(
                "terms", term_entry, "Each result should have a 'terms' field."
            )
            for term_data in term_entry["terms"]:
                self.assertIn(
                    "term", term_data, "Each term entry should have a 'term'."
                )
                self.assertIn(
                    "score", term_data, "Each term entry should have a 'score'."
                )

    def test_rake_extraction_strings(self):
        """Test RAKE extraction on a plain list of strings."""
        # Step 1: Create a list of documents
        text_corpus = [
            "RAKE identifies key phrases from text efficiently.",
            "Rapid Automatic Keyword Extraction is simple and effective.",
        ]

        # Extract terms using RAKE
        results = self.extractor.extract_terms_strings(text_corpus)

        # Assert results
        self.assertIsNotNone(results.terms, "Terms should not be None.")
        self.assertTrue(
            all("doc_id" in item for item in results.terms),
            "Each result should have 'doc_id'.",
        )

        for term_entry in results.terms:
            self.assertIn(
                "terms", term_entry, "Each result should have a 'terms' field."
            )
            for term_data in term_entry["terms"]:
                self.assertIn(
                    "term", term_data, "Each term entry should have a 'term'."
                )
                self.assertIn(
                    "score", term_data, "Each term entry should have a 'score'."
                )

    def test_rake_with_custom_stoplist_and_delimiters(self):
        """Test RAKE extraction with a custom stoplist and delimiters."""
        custom_stoplist = ["rapid", "automatic", "extraction"]
        custom_delimiters = [".", "!"]

        # Create a new RAKE extractor with the custom stoplist and delimiters
        custom_extractor = RAKETermExtractor(
            stoplist=custom_stoplist, phrase_delimiters=custom_delimiters, threshold=0.5
        )

        text_corpus = ["Rapid Automatic Keyword Extraction works efficiently!"]

        results = custom_extractor.extract_terms_strings(text_corpus)

        # Assert results
        self.assertIsNotNone(results.terms, "Terms should not be None.")
        for term_entry in results.terms:
            for term_data in term_entry["terms"]:
                # Ensure no term contains the custom stop words
                self.assertFalse(
                    any(word in term_data["term"] for word in custom_stoplist),
                    "Stop words found in terms.",
                )

    def test_rake_with_threshold(self):
        """Test RAKE extraction with a threshold to filter terms."""
        text_corpus = ["RAKE identifies key phrases and filters low scoring terms."]

        # Extract terms with a higher threshold
        self.extractor.threshold = 2.0
        results = self.extractor.extract_terms_strings(text_corpus)

        # Assert results
        self.assertIsNotNone(results.terms, "Terms should not be None.")
        for term_entry in results.terms:
            for term_data in term_entry["terms"]:
                self.assertGreaterEqual(
                    term_data["score"], 2.0, "Term score should meet the threshold."
                )

    def test_rake_with_no_threshold(self):
        """Test RAKE extraction without a threshold."""
        text_corpus = ["RAKE identifies all terms without filtering by score."]
        extractor_no_threshold = RAKETermExtractor(
            stoplist=self.stoplist, phrase_delimiters=self.delimiters, threshold=None
        )

        results = extractor_no_threshold.extract_terms_strings(text_corpus)

        # Assert results
        self.assertIsNotNone(results.terms, "Terms should not be None.")
        self.assertGreater(
            len(results.terms[0]["terms"]),
            0,
            "Should extract all terms without filtering.",
        )


# The example of Rose et al. (2010), Figure 1.1, without its stopwords and punctuation
STOPLIST = ["of", "over", "the", "a", "and", "for", "are", "these", "is", "again"]
DELIMITERS = list(".,;:!?()[]\"'")


def rake_scores(text, stoplist=STOPLIST, delimiters=DELIMITERS):
    results = RAKETermExtractor(stoplist, delimiters).extract_terms_strings([text])
    return {t["term"]: t["score"] for t in results.terms[0]["terms"]}


def test_candidates_split_at_stopwords():
    extractor = RAKETermExtractor(STOPLIST, DELIMITERS)
    assert extractor.generate_candidate_keywords(
        "Compatibility of systems of linear constraints over the set of natural "
        "numbers. Criteria (of compatibility) are given."
    ) == [
        ["compatibility"],
        ["systems"],
        ["linear", "constraints"],
        ["set"],
        ["natural", "numbers"],
        ["criteria"],
        ["compatibility"],
        ["given"],
    ]


def test_tokenize():
    extractor = RAKETermExtractor(STOPLIST, [",", "--"])
    assert extractor.tokenize("Heart--failure,  the  Pump") == [
        "heart",
        "--",
        "failure",
        ",",
        "the",
        "pump",
    ]
    assert RAKETermExtractor(STOPLIST, []).tokenize("A b.") == ["a", "b."]


def test_scores():
    # Degree is the number of words of the candidates a word occurs in, itself
    # included, so "linear" has degree 2 and "compatibility" 2 over 2 occurrences
    assert rake_scores(
        "Compatibility of systems of linear constraints over the set of natural "
        "numbers. Criteria of compatibility are given."
    ) == {
        "linear constraints": 4.0,
        "natural numbers": 4.0,
        "compatibility": 1.0,
        "systems": 1.0,
        "set": 1.0,
        "criteria": 1.0,
        "given": 1.0,
    }


def test_linked_keywords():
    # "axis" and "evil" adjoin with only stopwords between them twice, so are joined
    assert rake_scores("The axis of evil. Again the axis of evil.") == {
        "axis of evil": 2.0,
        "axis": 1.0,
        "evil": 1.0,
    }
    # Only once, so not joined
    assert "axis of evil" not in rake_scores("The axis of evil. The axis. The evil.")
    # A delimiter between them, so not joined
    assert "axis of evil" not in rake_scores("Axis, of evil. Axis, of evil.")


if __name__ == "__main__":
    unittest.main()
