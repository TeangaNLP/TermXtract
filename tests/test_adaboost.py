import unittest
from unittest import mock

from teanga import Corpus

from termxtract.supervised import adaboost
from termxtract.supervised.adaboost import AdaBoostTermExtractor, load_wordnet
from termxtract.supervised.term_extractor import SupervisedTermExtractor


class TestAdaBoostExtraction(unittest.TestCase):
    """Test cases for AdaBoost term extraction method."""

    def setUp(self):
        """Set up a small test corpus and labels."""
        self.corpus = Corpus()
        self.corpus.add_layer_meta("text")
        self.corpus.add_layer_meta("words", layer_type="span", base="text")

        doc1 = self.corpus.add_doc("Machine learning is widely used in AI research.")
        doc1.words = [(0, 7), (8, 16), (17, 19), (20, 25), (26, 30), (31, 33), (34, 42)]

        self.text_corpus = ["Machine learning is widely used in AI research."]

        # Labels (Supervised Learning)
        self.labels = {"machine learning": 1, "AI research": 1, "widely used": 0}

    def test_adaboost_extraction_teanga(self):
        """Test AdaBoost term extraction on a Teanga corpus."""
        extractor = SupervisedTermExtractor(method="adaboost", threshold=0.1, n=2)
        results = extractor.extract(self.corpus, self.labels)

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

    def test_adaboost_extraction_strings(self):
        """Test AdaBoost term extraction on a list of strings."""
        extractor = SupervisedTermExtractor(method="adaboost", threshold=0.1, n=2)
        results = extractor.extract(self.text_corpus, self.labels)

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

    def test_semantic_score_lemmatizes_and_handles_multiword_terms(self):
        """Test that inflected forms and multi-word terms are found in WordNet."""
        extractor = AdaBoostTermExtractor()
        self.assertGreater(extractor.compute_semantic_score("dogs"), 0.0)
        self.assertGreater(
            extractor.compute_semantic_score("artificial intelligence"), 0.0
        )
        self.assertEqual(extractor.compute_semantic_score("of machine"), 0.0)


class TestWordNetLoading(unittest.TestCase):
    """Test cases for lazy loading and downloading of the WordNet lexicon."""

    def test_lexicon_not_loaded_at_construction(self):
        """Test that constructing the extractor does not load or download WordNet."""
        with mock.patch.object(adaboost, "load_wordnet") as load:
            extractor = SupervisedTermExtractor(
                method="adaboost", wordnet_lexicon="oewn:2024"
            )
            load.assert_not_called()
            extractor.extractor.compute_semantic_score("dog")
            extractor.extractor.compute_semantic_score("cat")
            load.assert_called_once_with("oewn:2024")

    def test_installed_lexicon_is_not_downloaded(self):
        """Test that an installed lexicon is used without downloading."""
        with (
            mock.patch.object(adaboost.wn, "lexicons", return_value=["installed"]),
            mock.patch.object(adaboost.wn, "download") as download,
            mock.patch.object(adaboost.wn, "Wordnet") as wordnet,
        ):
            load_wordnet("oewn:2025+")
        download.assert_not_called()
        wordnet.assert_called_once()

    def test_missing_lexicon_is_downloaded(self):
        """Test that a missing lexicon is downloaded on first use."""
        with (
            mock.patch.object(adaboost.wn, "lexicons", return_value=[]),
            mock.patch.object(adaboost.wn, "download") as download,
            mock.patch.object(adaboost.wn, "Wordnet"),
        ):
            load_wordnet("oewn:2025+")
        download.assert_called_once_with("oewn:2025+", progress_handler=None)

    def test_download_failure_raises_clear_error(self):
        """Test that a failed download (e.g. offline) explains how to install the lexicon."""
        with (
            mock.patch.object(adaboost.wn, "lexicons", return_value=[]),
            mock.patch.object(
                adaboost.wn, "download", side_effect=OSError("network unreachable")
            ),
            self.assertRaises(RuntimeError) as ctx,
        ):
            load_wordnet("oewn:2025+")
        self.assertIn("python -m wn download oewn:2025+", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, OSError)


if __name__ == "__main__":
    unittest.main()
