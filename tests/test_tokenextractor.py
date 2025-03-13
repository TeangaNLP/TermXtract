import unittest
from teanga import Corpus
from termxtract.supervised.term_extractor import SupervisedTermExtractor

class TestTokenClassificationExtraction(unittest.TestCase):
    """Test cases for token classification term extraction method."""
    
    def setUp(self):
        """Set up a small test corpus and labels."""
        # Create Teanga corpus
        self.corpus = Corpus()
        self.corpus.add_layer_meta("text")
        self.corpus.add_layer_meta("words", layer_type="span", base="text")
        
        # Add document to Teanga corpus
        doc1 = self.corpus.add_doc("Machine learning is widely used in AI research.")
        doc1.words = [(0, 7), (8, 16), (17, 19), (20, 25), (26, 30), (31, 33), (34, 42)]
        
        # Add another document for more testing
        doc2 = self.corpus.add_doc("Neural networks have transformed deep learning applications.")
        doc2.words = [(0, 6), (7, 15), (16, 20), (21, 32), (33, 37), (38, 46), (47, 60)]
        
        # List of strings corpus
        self.text_corpus = [
            "Machine learning is widely used in AI research.",
            "Neural networks have transformed deep learning applications."
        ]
        
        # Labels for supervised learning
        self.labels = {
            "machine learning": 1,
            "AI research": 1,
            "neural networks": 1,
            "deep learning": 1,
            "widely used": 0, 
            "applications": 0
        }
    
    def test_token_classification_extraction_teanga(self):
        """Test token classification term extraction on a Teanga corpus."""
        extractor = SupervisedTermExtractor(
            method="token-classification", 
            model_name="xlm-roberta-base",
            n=2,
            batch_size=4,
            epochs=1  # Use single epoch for faster tests
        )
        
        # Test extraction (includes training)
        results = extractor.extract(self.corpus, self.labels)
        
        # Verify results format
        self.assertIsNotNone(results.terms, "Terms should not be None.")
        self.assertTrue(all("doc_id" in item for item in results.terms), "Each result should have 'doc_id'.")
        
        for term_entry in results.terms:
            self.assertIn("terms", term_entry, "Each result should have a 'terms' field.")
            for term_data in term_entry["terms"]:
                self.assertIn("term", term_data, "Each term entry should have a 'term'.")
                self.assertIn("score", term_data, "Each term entry should have a 'score'.")
    
    def test_token_classification_extraction_strings(self):
        """Test token classification term extraction on a list of strings."""
        extractor = SupervisedTermExtractor(
            method="token-classification", 
            model_name="xlm-roberta-base",
            n=2,
            batch_size=4,
            epochs=1  # Use single epoch for faster tests
        )
        
        # Test extraction (includes training)
        results = extractor.extract(self.text_corpus, self.labels)
        
        # Verify results format
        self.assertIsNotNone(results.terms, "Terms should not be None.")
        self.assertTrue(all("doc_id" in item for item in results.terms), "Each result should have 'doc_id'.")
        
        for term_entry in results.terms:
            self.assertIn("terms", term_entry, "Each result should have a 'terms' field.")
            for term_data in term_entry["terms"]:
                self.assertIn("term", term_data, "Each term entry should have a 'term'.")
                self.assertIn("score", term_data, "Each term entry should have a 'score'.")
    
    def test_token_classification_training_and_extraction_teanga(self):
        """Test separate training and extraction steps on Teanga corpus."""
        extractor = SupervisedTermExtractor(
            method="token-classification", 
            model_name="xlm-roberta-base",
            n=2,
            batch_size=4,
            epochs=1
        )
        
        # Get the underlying token classification extractor
        token_extractor = extractor.extractor
        
        # Train on first document only
        doc1_corpus = Corpus()
        doc1_corpus.add_layer_meta("text")
        doc1_corpus.add_layer_meta("words", layer_type="span", base="text")
        doc = doc1_corpus.add_doc("Machine learning is widely used in AI research.")
        doc.words = [(0, 7), (8, 16), (17, 19), (20, 25), (26, 30), (31, 33), (34, 42)]
        
        # Train the model with doc1
        train_results = token_extractor.extract_terms_teanga(doc1_corpus, self.labels)
        self.assertTrue(token_extractor.is_trained, "Model should be trained after training step")
        
        # Test extraction on doc2 (already trained model)
        doc2_corpus = Corpus()
        doc2_corpus.add_layer_meta("text")
        doc2_corpus.add_layer_meta("words", layer_type="span", base="text")
        doc = doc2_corpus.add_doc("Neural networks have transformed deep learning applications.")
        doc.words = [(0, 6), (7, 15), (16, 20), (21, 32), (33, 37), (38, 46), (47, 60)]
        
        # Extract terms from doc2
        extract_results = token_extractor.extract_terms_teanga(doc2_corpus)
        
        # Verify extraction results
        self.assertIsNotNone(extract_results.terms, "Extraction results should not be None")
        self.assertEqual(len(extract_results.terms), 1, "Should have results for one document")
    
    def test_token_classification_training_and_extraction_strings(self):
        """Test separate training and extraction steps on string lists."""
        extractor = SupervisedTermExtractor(
            method="token-classification", 
            model_name="xlm-roberta-base",
            n=2,
            batch_size=4,
            epochs=1
        )
        
        # Get the underlying token classification extractor
        token_extractor = extractor.extractor
        
        # Train on first document only
        train_corpus = ["Machine learning is widely used in AI research."]
        
        # Train the model 
        train_results = token_extractor.extract_terms_strings(train_corpus, self.labels)
        self.assertTrue(token_extractor.is_trained, "Model should be trained after training step")
        
        # Test extraction on second document (already trained model)
        extract_corpus = ["Neural networks have transformed deep learning applications."]
        
        # Extract terms
        extract_results = token_extractor.extract_terms_strings(extract_corpus)
        
        # Verify extraction results
        self.assertIsNotNone(extract_results.terms, "Extraction results should not be None")
        self.assertEqual(len(extract_results.terms), 1, "Should have results for one document")
    
    def test_token_classification_with_different_models(self):
        """Test term extraction with different transformer models."""
        # Test with another model type (smaller for faster tests)
        extractor = SupervisedTermExtractor(
            method="token-classification", 
            model_name="distilbert-base-uncased",  # Different model
            n=2,
            batch_size=4,
            epochs=1
        )
        
        # Run extraction
        results = extractor.extract(self.text_corpus, self.labels)
        
        # Verify results
        self.assertIsNotNone(results.terms, "Terms should not be None.")
        self.assertEqual(len(results.terms), len(self.text_corpus), 
                         "Number of results should match number of input documents")


if __name__ == "__main__":
    unittest.main()