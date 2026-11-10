import importlib.util
import unittest

from teanga import Corpus

from termxtract.supervised.term_extractor import SupervisedTermExtractor

HAS_TRANSFORMERS = all(
    importlib.util.find_spec(module) is not None
    for module in ("torch", "transformers", "sacremoses")
)


@unittest.skipUnless(HAS_TRANSFORMERS, "requires the optional 'transformers' extra")
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
        doc2 = self.corpus.add_doc(
            "Neural networks have transformed deep learning applications."
        )
        doc2.words = [(0, 6), (7, 15), (16, 20), (21, 32), (33, 37), (38, 46), (47, 60)]

        # List of strings corpus
        self.text_corpus = [
            "Machine learning is widely used in AI research.",
            "Neural networks have transformed deep learning applications.",
        ]

        # Labels for supervised learning
        self.labels = {
            "machine learning": 1,
            "AI research": 1,
            "neural networks": 1,
            "deep learning": 1,
            "widely used": 0,
            "applications": 0,
        }

    def test_token_classification_extraction_teanga(self):
        """Test token classification term extraction on a Teanga corpus."""
        extractor = SupervisedTermExtractor(
            method="token-classification",
            model_name="xlm-roberta-base",
            n=2,
            batch_size=4,
            epochs=1,  # Use single epoch for faster tests
        )

        # Test extraction (includes training)
        results = extractor.extract(self.corpus, self.labels)

        # Verify results format
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

    def test_token_classification_extraction_strings(self):
        """Test token classification term extraction on a list of strings."""
        extractor = SupervisedTermExtractor(
            method="token-classification",
            model_name="xlm-roberta-base",
            n=2,
            batch_size=4,
            epochs=1,  # Use single epoch for faster tests
        )

        # Test extraction (includes training)
        results = extractor.extract(self.text_corpus, self.labels)

        # Verify results format
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

    def test_token_classification_training_and_extraction_teanga(self):
        """Test separate training and extraction steps on Teanga corpus."""
        extractor = SupervisedTermExtractor(
            method="token-classification",
            model_name="xlm-roberta-base",
            n=2,
            batch_size=4,
            epochs=1,
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
        token_extractor.extract_terms_teanga(doc1_corpus, self.labels)
        self.assertTrue(
            token_extractor.is_trained, "Model should be trained after training step"
        )

        # Test extraction on doc2 (already trained model)
        doc2_corpus = Corpus()
        doc2_corpus.add_layer_meta("text")
        doc2_corpus.add_layer_meta("words", layer_type="span", base="text")
        doc = doc2_corpus.add_doc(
            "Neural networks have transformed deep learning applications."
        )
        doc.words = [(0, 6), (7, 15), (16, 20), (21, 32), (33, 37), (38, 46), (47, 60)]

        # Extract terms from doc2
        extract_results = token_extractor.extract_terms_teanga(doc2_corpus)

        # Verify extraction results
        self.assertIsNotNone(
            extract_results.terms, "Extraction results should not be None"
        )
        self.assertEqual(
            len(extract_results.terms), 1, "Should have results for one document"
        )

    def test_token_classification_training_and_extraction_strings(self):
        """Test separate training and extraction steps on string lists."""
        extractor = SupervisedTermExtractor(
            method="token-classification",
            model_name="xlm-roberta-base",
            n=2,
            batch_size=4,
            epochs=1,
        )

        # Get the underlying token classification extractor
        token_extractor = extractor.extractor

        # Train on first document only
        train_corpus = ["Machine learning is widely used in AI research."]

        # Train the model
        token_extractor.extract_terms_strings(train_corpus, self.labels)
        self.assertTrue(
            token_extractor.is_trained, "Model should be trained after training step"
        )

        # Test extraction on second document (already trained model)
        extract_corpus = [
            "Neural networks have transformed deep learning applications."
        ]

        # Extract terms
        extract_results = token_extractor.extract_terms_strings(extract_corpus)

        # Verify extraction results
        self.assertIsNotNone(
            extract_results.terms, "Extraction results should not be None"
        )
        self.assertEqual(
            len(extract_results.terms), 1, "Should have results for one document"
        )

    def test_fit_on_one_corpus_and_predict_on_another(self):
        """A model trained on one corpus tags another, with offsets for Teanga."""
        extractor = SupervisedTermExtractor(
            method="token-classification", n=2, batch_size=4, epochs=1
        ).fit(self.text_corpus[:1], self.labels)
        corpus = Corpus()
        corpus.add_layer_meta("text")
        corpus.add_layer_meta("words", layer_type="span", base="text")
        doc = corpus.add_doc("Neural networks need machine learning.")
        doc.words = [(0, 6), (7, 15), (16, 20), (21, 28), (29, 37), (37, 38)]
        results = extractor.predict(corpus)
        self.assertEqual(len(results.terms), 1)
        self.assertEqual(len(results.offsets), 1)
        terms = {term["term"] for term in results.terms[0]["terms"]}
        for occurrence in results.offsets[0]["offsets"]:
            start, end = occurrence["offset"]
            self.assertEqual(doc.text[start:end].lower(), occurrence["term"])
            self.assertIn(occurrence["term"], terms)

    def test_training_writes_nothing_to_the_working_directory(self):
        """Checkpoints and logs are not saved in the working directory (#68)."""
        import os
        import tempfile

        extractor = SupervisedTermExtractor(
            method="token-classification",
            model_name="xlm-roberta-base",
            n=2,
            batch_size=4,
            epochs=1,
        )
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            try:
                extractor.extract(self.text_corpus, self.labels)
                self.assertEqual(os.listdir(directory), [])
            finally:
                os.chdir(cwd)

    def test_fit_on_gold_terms_with_validation_and_save(self):
        """Training on gold spans with a validation set and a seed, then saving."""
        import tempfile

        corpus = Corpus()
        corpus.add_layer_meta("text")
        corpus.add_layer_meta("words", layer_type="span", base="text")
        corpus.add_layer_meta("terms", layer_type="span", base="text")
        for doc, terms in zip(self.corpus.docs, [[(0, 16)], [(0, 15), (33, 46)]]):
            new = corpus.add_doc(doc.text)
            new.words = [(s[0], s[1]) for s in doc.words]
            new.terms = terms

        def train():
            return SupervisedTermExtractor(
                method="token-classification",
                model_name="distilbert-base-uncased",
                batch_size=4,
                epochs=3,
                random_state=0,
                patience=1,
            ).fit(corpus, validation=corpus)

        extractor = train()
        results = extractor.predict(corpus)
        self.assertEqual(results.terms, train().predict(corpus).terms)
        with tempfile.TemporaryDirectory() as directory:
            extractor.save(directory)
            loaded = SupervisedTermExtractor(
                method="token-classification", model_name=directory
            )
            self.assertTrue(loaded.extractor.is_trained)
            self.assertEqual(loaded.predict(corpus).terms, results.terms)

    def test_patience_needs_a_validation_set(self):
        extractor = SupervisedTermExtractor(
            method="token-classification",
            model_name="distilbert-base-uncased",
            patience=1,
        )
        with self.assertRaisesRegex(ValueError, "validation set"):
            extractor.fit(self.text_corpus, self.labels)

    def test_token_classification_with_different_models(self):
        """Test term extraction with different transformer models."""
        # Test with another model type (smaller for faster tests)
        extractor = SupervisedTermExtractor(
            method="token-classification",
            model_name="distilbert-base-uncased",  # Different model
            n=2,
            batch_size=4,
            epochs=1,
        )

        # Run extraction
        results = extractor.extract(self.text_corpus, self.labels)

        # Verify results
        self.assertIsNotNone(results.terms, "Terms should not be None.")
        self.assertEqual(
            len(results.terms),
            len(self.text_corpus),
            "Number of results should match number of input documents",
        )


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAS_TRANSFORMERS, "requires the optional 'transformers' extra")
class TestTokenClassificationTrainingData(unittest.TestCase):
    """Test how labels become token tags, without loading a model."""

    def test_only_positive_labels_are_tagged(self):
        from sacremoses import MosesDetokenizer

        from termxtract.supervised.tokenextractor import (
            TokenClassificationTermExtractor,
        )

        extractor = object.__new__(TokenClassificationTermExtractor)
        extractor.n = 2
        extractor.moses_detokenizer = MosesDetokenizer(lang="en")
        tokens = ["Machine", "learning", "is", "widely", "used", "in", "AI", "research"]
        labels = {"machine learning": 1, "AI research": 1, "widely used": 0}

        [(_, tags)] = extractor.create_training_data([(tokens, "")], labels)

        self.assertEqual(tags, ["B", "I", "O", "O", "O", "O", "B", "I"])

    def test_longest_labelled_term_is_tagged(self):
        from sacremoses import MosesDetokenizer

        from termxtract.supervised.tokenextractor import (
            TokenClassificationTermExtractor,
        )

        extractor = object.__new__(TokenClassificationTermExtractor)
        extractor.n = 3
        extractor.moses_detokenizer = MosesDetokenizer(lang="en")
        tokens = ["acute", "heart", "failure", "and", "heart", "rate"]
        labels = {"heart": 1, "heart failure": 1, "acute heart failure": 1}

        [(_, tags)] = extractor.create_training_data([(tokens, "")], labels)

        self.assertEqual(tags, ["B", "I", "I", "O", "B", "O"])

    def test_gold_spans_are_tagged_longest_first(self):
        from termxtract.supervised.tokenextractor import (
            TokenClassificationTermExtractor,
        )

        extractor = object.__new__(TokenClassificationTermExtractor)
        # "Acute heart failure. Heart rate rises."
        sentences = [
            (["Acute", "heart", "failure", "."], [(0, 5), (6, 11), (12, 19), (19, 20)]),
            (["Heart", "rate", "rises", "."], [(21, 26), (27, 31), (32, 37), (37, 38)]),
        ]
        # The longest term runs across the sentences, so the second part starts with
        # B. The terms it overlaps are not tagged, but "heart" is, and (22, 24)
        # covers no whole word.
        terms = [(6, 11), (0, 19), (6, 19), (12, 31), (22, 24)]

        data = extractor.create_span_training_data(sentences, terms)

        self.assertEqual(
            [tags for _, tags in data],
            [["O", "B", "B", "I"], ["B", "I", "O", "O"]],
        )
        terms = [(6, 11), (0, 19), (6, 19), (21, 31)]
        data = extractor.create_span_training_data(sentences, terms)
        self.assertEqual(
            [tags for _, tags in data],
            [["B", "I", "I", "O"], ["B", "I", "O", "O"]],
        )

    def test_sentences_layer(self):
        from termxtract.supervised.tokenextractor import (
            TokenClassificationTermExtractor,
        )

        corpus = Corpus()
        corpus.add_layer_meta("text")
        corpus.add_layer_meta("words", layer_type="span", base="text")
        corpus.add_layer_meta("sentences", layer_type="span", base="text")
        doc = corpus.add_doc("Dr. Smith treats heart failure. It is common")
        doc.words = [(0, 3), (4, 9), (10, 16), (17, 22), (23, 30), (30, 31)] + [
            (32, 34),
            (35, 37),
            (38, 44),
        ]
        doc.sentences = [(0, 31), (32, 44)]
        extractor = object.__new__(TokenClassificationTermExtractor)
        extractor.words_layer = None
        extractor.sentences_layer = None

        [sentences] = extractor._teanga_sentences(corpus)

        # The layer gives the sentences, so "Dr." does not end one
        self.assertEqual(
            [tokens for tokens, _ in sentences],
            [
                ["Dr.", "Smith", "treats", "heart", "failure", "."],
                ["It", "is", "common"],
            ],
        )
        self.assertEqual(sentences[1][1], [(32, 34), (35, 37), (38, 44)])

    def test_validation_f1(self):
        import numpy as np
        from transformers import EvalPrediction

        from termxtract.supervised.tokenextractor import (
            TokenClassificationTermExtractor,
        )

        extractor = object.__new__(TokenClassificationTermExtractor)
        extractor.id_to_label = {0: "O", 1: "B", 2: "I"}
        # Gold terms (0, 2) and (3, 4); the model tags (0, 2) and (2, 3). The first
        # and last model tokens are ignored (-100), as are padding tokens.
        labels = np.array([[-100, 1, 2, 0, 1, -100, -100]])
        tags = [0, 1, 2, 1, 0, 1, 1]
        predictions = np.eye(3)[tags][np.newaxis]

        f1 = extractor._f1(EvalPrediction(predictions=predictions, label_ids=labels))

        self.assertEqual(f1, {"f1": 0.5})

    def test_each_document_lists_the_terms_tagged_in_it(self):
        from unittest import mock

        from sacremoses import MosesDetokenizer

        from termxtract.stopwords import load_stopwords
        from termxtract.supervised.tokenextractor import (
            TokenClassificationTermExtractor,
        )

        extractor = object.__new__(TokenClassificationTermExtractor)
        extractor.is_trained = True
        extractor.stopwords = load_stopwords("english")
        extractor.moses_detokenizer = MosesDetokenizer(lang="en")
        documents = [
            [(["Plant", "cells", "grow", "."], ""), (["Cells", "divide", "."], "")],
            [(["A", "cell", "wall", "."], "")],
        ]
        spans = [
            [[(0, 5), (6, 11), (12, 16), (16, 17)], [(18, 23), (24, 30), (30, 31)]],
            [[(0, 1), (2, 6), (7, 11), (11, 12)]],
        ]
        # The model tags "plant cells" and "cells" in document 0, "cell wall" in 1
        tags = [["B", "I", "O", "O"], ["B", "O", "O"], ["O", "B", "I", "O"]]
        with (
            mock.patch.object(extractor, "tokenize_and_align_labels"),
            mock.patch.object(extractor, "create_torch_dataset"),
            mock.patch.object(extractor, "predict_tags", return_value=tags),
        ):
            results = extractor.extract_documents(None, ["d0", "d1"], documents, spans)

        terms = {
            doc["doc_id"]: {t["term"] for t in doc["terms"]} for doc in results.terms
        }
        # "cell" is not listed for document 0, although "cells" contains it
        self.assertEqual(terms, {"d0": {"plant cells", "cells"}, "d1": {"cell wall"}})
        self.assertEqual(
            results.offsets[0]["offsets"],
            [
                {"term": "plant cells", "offset": (0, 11)},
                {"term": "cells", "offset": (18, 23)},
            ],
        )
