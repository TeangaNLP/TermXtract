import unittest

from teanga import Corpus

from termxtract.unsupervised.term_extractor import TermExtractor


def _make_teanga_corpus(texts):
    """Build a Teanga corpus with a `words` span layer from plain strings."""
    corpus = Corpus()
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("words", layer_type="span", base="text")
    for text in texts:
        doc = corpus.add_doc(text)
        start = 0
        offsets = []
        for word in text.split():
            offsets.append((start, start + len(word)))
            start += len(word) + 1
        doc.words = offsets
    return corpus


class TestReferenceCorpusTypeMismatch(unittest.TestCase):
    """The reference corpus must use the same representation as the target corpus.

    Regression test for https://github.com/TeangaNLP/TermXtract/issues/25: mixing a
    Teanga reference with a list-of-strings target (or vice versa) used to fail deep
    inside the extraction with an unhelpful AttributeError instead of a clear TypeError.
    """

    METHODS = ("weirdness", "relevance", "domainpertinence")

    def setUp(self):
        self.target_strings = [
            "Weirdness focuses on domain-specific terminology.",
            "Specific terms are more common in the target corpus.",
        ]
        self.reference_strings = [
            "General reference corpus contains common phrases and terms.",
            "Reference texts are not domain-specific.",
        ]
        self.target_teanga = _make_teanga_corpus(self.target_strings)
        self.reference_teanga = _make_teanga_corpus(self.reference_strings)

    def test_list_target_with_teanga_reference_raises_type_error(self):
        for method in self.METHODS:
            with self.subTest(method=method):
                extractor = TermExtractor(
                    method=method, reference_corpus=self.reference_teanga, threshold=0.1, n=2
                )
                with self.assertRaises(TypeError):
                    extractor.extract(self.target_strings)

    def test_teanga_target_with_list_reference_raises_type_error(self):
        for method in self.METHODS:
            with self.subTest(method=method):
                extractor = TermExtractor(
                    method=method, reference_corpus=self.reference_strings, threshold=0.1, n=2
                )
                with self.assertRaises(TypeError):
                    extractor.extract(self.target_teanga)

    def test_matching_types_still_extract(self):
        for method in self.METHODS:
            with self.subTest(method=method):
                strings_extractor = TermExtractor(
                    method=method, reference_corpus=self.reference_strings, threshold=0.1, n=2
                )
                self.assertIsNotNone(strings_extractor.extract(self.target_strings).terms)

                teanga_extractor = TermExtractor(
                    method=method, reference_corpus=self.reference_teanga, threshold=0.1, n=2
                )
                self.assertIsNotNone(teanga_extractor.extract(self.target_teanga).terms)


if __name__ == "__main__":
    unittest.main()
