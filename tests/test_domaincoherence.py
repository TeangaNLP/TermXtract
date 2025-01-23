class TestDomainCoherenceExtraction(unittest.TestCase):
    def test_domaincoherence_extraction_teanga(self):
        # Step 1: Create a Teanga corpus
        corpus = Corpus()
        corpus.add_layer_meta("text")
        corpus.add_layer_meta("words", layer_type="span", base="text")

        doc1 = corpus.add_doc("The term extraction algorithm uses domain coherence.")
        doc1.words = [(0, 3), (4, 8), (9, 19), (20, 28), (29, 33), (34, 40), (41, 50)]

        doc2 = corpus.add_doc("Domain coherence is calculated using term contexts.")
        doc2.words = [(0, 6), (7, 15), (16, 18), (19, 30), (31, 37), (38, 46)]

        # Step 2: Initialize TermExtractor with DomainCoherence
        extractor = TermExtractor(method="domaincoherence", threshold=0.1, n=2, window_size=5)

        # Step 3: Extract terms
        results = extractor.extract(corpus)

        self.assertIsNotNone(results.terms, "Terms should not be None.")
        self.assertTrue(all("doc_id" in item for item in results.terms), "Each result should have 'doc_id'.")
        for term_entry in results.terms:
            self.assertIn("terms", term_entry, "Each result should have a 'terms' field.")
            for term_data in term_entry["terms"]:
                self.assertIn("term", term_data, "Each term entry should have a 'term'.")
                self.assertIn("score", term_data, "Each term entry should have a 'score'.")

