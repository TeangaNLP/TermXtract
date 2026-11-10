import re
import tempfile
from typing import Any

import numpy as np
import torch
from sacremoses import MosesDetokenizer, MosesTokenizer
from teanga import Corpus
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from ..layers import word_spans
from ..stopwords import Stopwords, load_stopwords
from ..utils import ATEResults, rank_terms


class TokenClassificationTermExtractor:
    """Term extraction using transformer-based token classification models."""

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        n: int = 6,
        batch_size: int = 8,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        max_length: int = 512,
        language: str = "en",
        stopwords: Stopwords = "english",
        words_layer: str | None = None,
    ):
        """
        Initialize the Token Classification Term Extractor.

        Args:
            model_name (str): Hugging Face model to use for token classification.
            n (int): Maximum n-gram size to consider.
            batch_size (int): Batch size for training and evaluation.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for training.
            max_length (int): Maximum sequence length for input tokenization.
            language (str): Language code for tokenization (en, fr, nl).
            stopwords (Union[str, Iterable[str], None]): Drop extracted terms that start or
                end with a stopword: "english", a list of words, or None.
            words_layer (Optional[str]): For Teanga input, the span layer holding the
                words, or None for `words` or, failing that, `tokens`.
        """
        self.model_name = model_name
        self.n = n
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.language = language
        self.stopwords = load_stopwords(stopwords)
        self.words_layer = words_layer

        # Initialize tokenizers (for preprocessing)
        self.moses_tokenizer = MosesTokenizer(lang=language)
        self.moses_detokenizer = MosesDetokenizer(lang=language)

        # Label mapping for token classification
        self.label_list = [
            "n",
            "B-T",
            "T",
        ]  # non-term, term beginning, term continuation
        self.label_to_id = {label: i for i, label in enumerate(self.label_list)}
        self.id_to_label = dict(enumerate(self.label_list))

        # Initialize transformer model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=len(self.label_list)
        )

        # Initialize trainer
        self.trainer: Trainer | None = None
        self.is_trained = False

    def preprocess_text(self, text: str) -> list[tuple[list[str], str]]:
        """
        Split text into sentences and tokenize.

        Args:
            text (str): Input text to preprocess.

        Returns:
            List[Tuple[List[str], str]]: List of tuples (tokens, original sentence).
        """
        # Simple sentence splitting by common delimiters
        sentences = re.split(r"(?<=[.!?])\s+", text)

        sentence_list = []
        for sentence in sentences:
            if not sentence.strip():
                continue

            # Tokenize using Moses tokenizer
            tokenized_text = self.moses_tokenizer.tokenize(sentence, return_str=True)
            sentence_list.append((tokenized_text.split(), sentence))

        return sentence_list

    def find_sublist_indices(
        self, sublist: list[str], full_list: list[str]
    ) -> list[tuple[int, int]]:
        """
        Find all occurrences of a sublist within a larger list.

        Args:
            sublist (List[str]): The sublist to find.
            full_list (List[str]): The full list to search in.

        Returns:
            List[Tuple[int, int]]: List of (start, end) indices where the sublist occurs.
        """
        results = []
        sublist_len = len(sublist)

        for ind in (i for i, e in enumerate(full_list) if e == sublist[0]):
            if full_list[ind : ind + sublist_len] == sublist:
                results.append((ind, ind + sublist_len - 1))

        return results

    def create_training_data(
        self, sentence_list: list[tuple[list[str], str]], terms: dict[str, int]
    ) -> list[tuple[list[str], list[str]]]:
        """
        Create training data with token-level annotations.

        Args:
            sentence_list (List[Tuple[List[str], str]]): List of (tokens, original sentence) tuples.
            terms (Dict[str, int]): Dictionary with terms as keys and binary labels as values.

        Returns:
            List[Tuple[List[str], List[str]]]: List of (tokens, tags) tuples.
        """
        training_data = []
        # Only terms labelled 1 are tagged; labels are matched case-insensitively
        positive_terms = {term.lower() for term, label in terms.items() if label == 1}

        for sentence_tokens, original_sentence in sentence_list:
            if not sentence_tokens:
                continue

            # Initialize all tokens as non-terms
            tags = ["n"] * len(sentence_tokens)

            # Iterate through n-grams from 1 to n
            for i in range(1, min(self.n + 1, len(sentence_tokens) + 1)):
                # Create n-grams
                for start_idx in range(len(sentence_tokens) - i + 1):
                    n_gram_tokens = sentence_tokens[start_idx : start_idx + i]
                    n_gram_text = self.moses_detokenizer.detokenize(n_gram_tokens)

                    # Check if n-gram is in the term list
                    if n_gram_text.lower() in positive_terms:
                        # Mark the beginning token as B-T
                        tags[start_idx] = "B-T"

                        # Mark continuation tokens as T
                        for j in range(start_idx + 1, start_idx + i):
                            tags[j] = "T"

            training_data.append((sentence_tokens, tags))

        return training_data

    def tokenize_and_align_labels(
        self, texts: list[list[str]], tags: list[list[str]]
    ) -> dict:
        """
        Tokenize inputs and align labels for transformer model.

        Args:
            texts (List[List[str]]): List of tokenized sentences.
            tags (List[List[str]]): List of token-level tags.

        Returns:
            Dict: Tokenized inputs with aligned labels.
        """
        tokenized_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            is_split_into_words=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = []
        for i, label in enumerate(tags):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                # Special tokens have a word id that is None. Mark as -100 to be ignored in loss.
                if word_idx is None:
                    label_ids.append(-100)
                # First token of each word gets the tag
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label_to_id[label[word_idx]])
                # Subsequent tokens of a word are ignored (-100)
                else:
                    label_ids.append(-100)

                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def create_torch_dataset(self, tokenized_inputs: dict) -> torch.utils.data.Dataset:
        """
        Create a PyTorch dataset from tokenized inputs.

        Args:
            tokenized_inputs (Dict): Tokenized inputs with labels.

        Returns:
            torch.utils.data.Dataset: Dataset for training/evaluation.
        """

        class Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                return item

            def __len__(self):
                return len(self.labels)

        return Dataset(tokenized_inputs, tokenized_inputs["labels"])

    def extract_terms_from_predictions(
        self, token_predictions: list[list[str]], texts: list[list[str]]
    ) -> dict[str, float]:
        """
        Extract terms from token-level predictions.

        Args:
            token_predictions (List[List[str]]): Token-level predictions.
            texts (List[List[str]]): Original tokenized texts.

        Returns:
            Dict[str, float]: Dictionary of extracted terms with confidence scores.
        """
        # The model tags terms rather than scoring them, so every term scores 1.0
        return {
            term: 1.0
            for pred, txt in zip(token_predictions, texts)
            for term, _, _ in self.tagged_terms(pred, txt)
        }

    def tagged_terms(
        self, tags: list[str], tokens: list[str]
    ) -> list[tuple[str, int, int]]:
        """
        Find the terms tagged in a sentence: each `B-T` with the `T` tags after it.

        Args:
            tags (List[str]): The tag of each token.
            tokens (List[str]): The tokens of the sentence.

        Returns:
            List[Tuple[str, int, int]]: Each term in lower case, with the index of its
            first token and of the token after its last.
        """
        terms = []
        j = 0
        while j < len(tags):
            if tags[j] == "B-T":
                k = j + 1
                while k < len(tags) and tags[k] == "T":
                    k += 1
                term = self.moses_detokenizer.detokenize(tokens[j:k])
                terms.append((term.lower(), j, k))
                j = k
            else:
                j += 1
        return terms

    def train_model(self, tokenized_dataset: torch.utils.data.Dataset):
        """
        Train the token classification model.

        Args:
            tokenized_dataset: Training dataset.
        """
        # The trained model is kept in memory, so save no checkpoints or logs, and give
        # the Trainer a temporary directory rather than one in the working directory
        with tempfile.TemporaryDirectory() as output_dir:
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=self.epochs,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size * 2,
                warmup_steps=0,
                weight_decay=0.01,
                learning_rate=self.learning_rate,
                logging_steps=100,
                save_strategy="no",
                eval_strategy="no",
                report_to="none",
            )
            self.trainer = Trainer(
                model=self.model, args=training_args, train_dataset=tokenized_dataset
            )
            self.trainer.train()
        self.is_trained = True

    def predict_sentences(
        self, tokenized_dataset: torch.utils.data.Dataset, texts
    ) -> list[dict[str, float]]:
        """
        Make predictions using the trained model, for each sentence.

        Args:
            tokenized_dataset: Dataset for prediction.
            texts (List[List[str]]): Original tokenized texts.

        Returns:
            List[Dict[str, float]]: The terms extracted from each sentence, with
            confidence scores.
        """
        return [
            self.extract_terms_from_predictions([prediction], [text])
            for prediction, text in zip(self.predict_tags(tokenized_dataset), texts)
        ]

    def predict_tags(
        self, tokenized_dataset: torch.utils.data.Dataset
    ) -> list[list[str]]:
        """
        Tag the tokens of each sentence with the trained model.

        Args:
            tokenized_dataset: Dataset for prediction.

        Returns:
            List[List[str]]: The tag of each token of each sentence.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")

        if not self.trainer:
            with tempfile.TemporaryDirectory() as output_dir:
                self.trainer = Trainer(
                    model=self.model,
                    args=TrainingArguments(
                        output_dir=output_dir,
                        per_device_eval_batch_size=self.batch_size * 2,
                        report_to="none",
                    ),
                )

        # Get predictions
        predictions, labels, _ = self.trainer.predict(tokenized_dataset)
        predictions = np.argmax(predictions, axis=2)

        # Convert to label names, keeping the first model token of each word
        return [
            [
                self.id_to_label[p]
                for (p, gold) in zip(prediction, label)
                if gold != -100
            ]
            # The dataset always has (placeholder) labels, so `labels` is never None
            for prediction, label in zip(predictions, labels)  # type: ignore[arg-type]
        ]

    def predict(
        self, tokenized_dataset: torch.utils.data.Dataset, texts
    ) -> dict[str, float]:
        """
        Make predictions using the trained model.

        Args:
            tokenized_dataset: Dataset for prediction.
            texts (List[List[str]]): Original tokenized texts.

        Returns:
            Dict[str, float]: Dictionary of extracted terms with confidence scores.
        """
        term_scores: dict[str, float] = {}
        for sentence_terms in self.predict_sentences(tokenized_dataset, texts):
            term_scores.update(sentence_terms)
        return term_scores

    def prepare_teanga_documents(
        self, corpus: Corpus
    ) -> list[list[tuple[list[str], str]]]:
        """
        Split each document of a Teanga corpus into sentences.

        Args:
            corpus: Teanga corpus object.

        Returns:
            List[List[Tuple[List[str], str]]]: For each document, its (tokens, original
            sentence) tuples.
        """
        return [
            [(tokens, " ".join(tokens)) for tokens, _ in sentences]
            for sentences in self._teanga_sentences(corpus)
        ]

    def _teanga_sentences(
        self, corpus: Corpus
    ) -> list[list[tuple[list[str], list[tuple[int, int]]]]]:
        """
        Split each document of a Teanga corpus into sentences, after each word that is
        ".", "!" or "?".

        Returns:
            For each document, the words and the word spans of each sentence.
        """
        documents = []
        for doc_id in corpus.doc_ids:
            doc = corpus.doc_by_id(doc_id)
            sentences = []
            tokens: list[str] = []
            spans: list[tuple[int, int]] = []
            for start, end in word_spans(doc, self.words_layer):
                tokens.append(doc.text[start:end])
                spans.append((start, end))
                if tokens[-1] in (".", "!", "?"):
                    sentences.append((tokens, spans))
                    tokens, spans = [], []
            if tokens:
                sentences.append((tokens, spans))
            documents.append(sentences)
        return documents

    def prepare_teanga_corpus(self, corpus: Corpus) -> list[tuple[list[str], str]]:
        """
        Prepare a Teanga corpus for term extraction.

        Args:
            corpus: Teanga corpus object.

        Returns:
            List[Tuple[List[str], str]]: List of (tokens, original sentence) tuples.
        """
        return [
            sentence
            for sentences in self.prepare_teanga_documents(corpus)
            for sentence in sentences
        ]

    def extract_documents(
        self,
        corpus: Corpus | list[str],
        doc_ids: list[str],
        documents: list[list[tuple[list[str], str]]],
        spans: list[list[list[tuple[int, int]]]] | None = None,
    ) -> ATEResults:
        """
        Tag the sentences of each document with the trained model.

        Each document lists the terms the model tagged in its own sentences.

        Args:
            corpus (Union[Corpus, List[str]]): The input corpus, returned in the results.
            doc_ids (List[str]): The ID of each document.
            documents (List[List[Tuple[List[str], str]]]): For each document, its
                (tokens, original sentence) tuples.
            spans (Optional[List[List[List[Tuple[int, int]]]]]): The character offsets
                of each token of each sentence of each document, to give the offsets of
                the tagged terms, or None.

        Returns:
            ATEResults: Results object with extracted terms, and their offsets if
            `spans` is given.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")

        # Tag every sentence, with dummy labels (all "n") for the tokenizer
        texts = [tokens for sentences in documents for tokens, _ in sentences]
        tags = [["n"] * len(tokens) for tokens in texts]
        tokenized_inputs = self.tokenize_and_align_labels(texts, tags)
        tokenized_dataset = self.create_torch_dataset(tokenized_inputs)
        sentence_tags = iter(self.predict_tags(tokenized_dataset))

        terms_by_doc = []
        offsets_by_doc = []
        for k, (doc_id, sentences) in enumerate(zip(doc_ids, documents)):
            # The model tags terms rather than scoring them, so every term scores 1.0
            found = [
                (term, s, start, end)
                for s, (tokens, _) in enumerate(sentences)
                for term, start, end in self.tagged_terms(next(sentence_tags), tokens)
            ]
            doc_terms: dict[str, Any] = {
                "doc_id": doc_id,
                "terms": rank_terms(
                    dict.fromkeys((term for term, _, _, _ in found), 1.0),
                    stopwords=self.stopwords,
                ),
            }
            terms_by_doc.append(doc_terms)
            if spans is not None:
                kept = {term["term"] for term in doc_terms["terms"]}
                offsets_by_doc.append(
                    {
                        "doc_id": doc_id,
                        "offsets": [
                            {
                                "term": term,
                                "offset": (
                                    spans[k][s][start][0],
                                    spans[k][s][end - 1][1],
                                ),
                            }
                            for term, s, start, end in found
                            if term in kept
                        ],
                    }
                )

        return ATEResults(
            corpus=corpus,
            terms=terms_by_doc,
            offsets=offsets_by_doc if spans is not None else None,
        )

    def fit_strings(self, corpus: list[str], labels: dict[str, int]) -> None:
        """
        Fine-tune the model on a list of strings, tagging the terms labelled 1.

        Args:
            corpus (List[str]): List of text documents.
            labels (Dict[str, int]): Dictionary with terms as keys and binary labels as values.
        """
        self._fit(
            [sentence for doc in corpus for sentence in self.preprocess_text(doc)],
            labels,
        )

    def fit_teanga(self, corpus: Corpus, labels: dict[str, int]) -> None:
        """
        Fine-tune the model on a Teanga corpus, tagging the terms labelled 1.

        Args:
            corpus: Teanga corpus object.
            labels (Dict[str, int]): Dictionary with terms as keys and binary labels as values.
        """
        self._fit(self.prepare_teanga_corpus(corpus), labels)

    def _fit(
        self, sentence_list: list[tuple[list[str], str]], labels: dict[str, int]
    ) -> None:
        training_data = self.create_training_data(sentence_list, labels)
        train_texts = [tokens for tokens, _ in training_data]
        train_tags = [tags for _, tags in training_data]
        train_tokenized = self.tokenize_and_align_labels(train_texts, train_tags)
        self.train_model(self.create_torch_dataset(train_tokenized))

    def predict_strings(self, corpus: list[str]) -> ATEResults:
        """
        Tag the terms of a list of strings with the trained model.

        Args:
            corpus (List[str]): List of text documents.

        Returns:
            ATEResults: The terms tagged in each document.
        """
        return self.extract_documents(
            corpus,
            [f"doc_{i}" for i in range(len(corpus))],
            [self.preprocess_text(doc) for doc in corpus],
        )

    def predict_teanga(self, corpus: Corpus) -> ATEResults:
        """
        Tag the terms of a Teanga corpus with the trained model.

        Args:
            corpus: Teanga corpus object.

        Returns:
            ATEResults: The terms tagged in each document, with their offsets.
        """
        sentences = self._teanga_sentences(corpus)
        return self.extract_documents(
            corpus,
            list(corpus.doc_ids),
            [[(tokens, " ".join(tokens)) for tokens, _ in doc] for doc in sentences],
            [[spans for _, spans in doc] for doc in sentences],
        )

    def extract_terms_teanga(
        self, corpus: Corpus, labels: dict[str, int] | None = None
    ) -> ATEResults:
        """
        Extract terms from a Teanga corpus, training on it first if labels are given.

        Args:
            corpus: Teanga corpus object.
            labels (Optional[Dict[str, int]]): Dictionary with terms as keys and binary
                labels as values, to train on, or None to use the trained model.

        Returns:
            ATEResults: Results object with extracted terms.
        """
        if labels is not None:
            self.fit_teanga(corpus, labels)
        return self.predict_teanga(corpus)

    def train_terms_teanga(self, corpus: Corpus, labels: dict[str, int]) -> ATEResults:
        """
        Train the model and extract terms from a Teanga corpus.

        Args:
            corpus: Teanga corpus object.
            labels (Dict[str, int]): Dictionary with terms as keys and binary labels as values.

        Returns:
            ATEResults: Results object with extracted terms.
        """
        return self.extract_terms_teanga(corpus, labels)

    def extract_terms_strings(
        self, corpus: list[str], labels: dict[str, int] | None = None
    ) -> ATEResults:
        """
        Extract terms from a list of strings, training on it first if labels are given.

        Args:
            corpus (List[str]): List of text documents.
            labels (Optional[Dict[str, int]]): Dictionary with terms as keys and binary
                labels as values, to train on, or None to use the trained model.

        Returns:
            ATEResults: Results object with extracted terms.
        """
        if labels is not None:
            self.fit_strings(corpus, labels)
        return self.predict_strings(corpus)

    def train_terms_strings(
        self, corpus: list[str], labels: dict[str, int]
    ) -> ATEResults:
        """
        Train the model and extract terms from a list of strings.

        Args:
            corpus (List[str]): List of text documents.
            labels (Dict[str, int]): Dictionary with terms as keys and binary labels as values.

        Returns:
            ATEResults: Results object with extracted terms.
        """
        return self.extract_terms_strings(corpus, labels)
