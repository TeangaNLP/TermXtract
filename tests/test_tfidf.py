import unittest
from termxtract.tfidf import TFIDFTermExtractor

class TestTFIDFTermExtractor(unittest.TestCase):
    
    def test_tfidf(self):
        corpus = [
            "this is a sample document",
            "this document is another sample",
            "sample document with different terms"
        ]
        extractor = TFIDFTermExtractor()
        tfidf_scores = extractor.compute_tfidf(corpus)

        # Ensure there are 3 documents with tf-idf scores
        self.assertEqual(len(tfidf_scores), 3)
        
        # Check that 'sample' and 'document' exist in the TF-IDF scores
        self.assertIn('sample', tfidf_scores[0])
        self.assertIn('document', tfidf_scores[1])

if __name__ == '__main__':
    unittest.main()

