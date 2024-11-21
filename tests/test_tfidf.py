import unittest
from teanga import Corpus
from termxtract.term_extractor import TermExtractor

class TestTFIDFExample(unittest.TestCase):

    def test_tfidf_extraction_example(self):
        # Step 1: Create a Teanga corpus and add layers
        corpus = Corpus()
        corpus.add_layer_meta("text")
        corpus.add_layer_meta("words", layer_type="span", base="text")

        # Step 2: Add documents with word offsets
        doc1 = corpus.add_doc("This is an example document.")
        doc1.words = [(0, 4), (5, 7), (8, 10), (11, 18), (19, 28)]

        doc2 = corpus.add_doc("TFIDF extraction with unigrams, bigrams, and trigrams.")
        doc2.words = [(0, 5), (6, 16), (17, 21), (22, 30), (31, 39), (40, 44), (45, 53)]

        # Step 3: Initialize TermExtractor with TF-IDF, n=3 for 1-gram, 2-gram, and 3-gram extraction
        extractor = TermExtractor(method="tfidf", threshold=0.1, n=3)

        # Step 4: Perform term extraction on the corpus
        terms_offsets_by_doc = extractor.extract(corpus)

        # Step 5: Print results to demonstrate output format
        print("TF-IDF Extraction Results with n-grams (1, 2, 3):")
        for doc_id, terms_offsets in terms_offsets_by_doc.items():
            print(f"\nDocument ID: {doc_id}")
            for term, offsets in terms_offsets.items():
                print(f"Term: '{term}', Offsets: {offsets}")

        # Example assertions to validate output structure (for testing purposes)
        for doc_id, terms_offsets in terms_offsets_by_doc.items():
            self.assertTrue(isinstance(terms_offsets, dict), "Expected terms_offsets to be a dictionary.")
            for term, offsets in terms_offsets.items():
                self.assertTrue(isinstance(term, str), "Each term should be a string.")
                self.assertTrue(all(isinstance(offset, tuple) and len(offset) == 2 for offset in offsets),
                                "Each offset should be a tuple (start, end).")

if __name__ == '__main__':
    unittest.main()
