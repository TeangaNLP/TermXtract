import math
import re
import tempfile
from bisect import bisect_left
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sacremoses import MosesDetokenizer, MosesTokenizer
from teanga import Corpus
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)

from ..datasets.base import GOLD_LAYER, SENTENCES_LAYER
from ..layers import find_layer, word_annotations, word_spans
from ..stopwords import Stopwords, load_stopwords
from ..utils import ATEResults, rank_terms


def tag_spans(tags: list[str]) -> list[tuple[int, int]]:
    """
    Find the terms in a sequence of tags: each `B` with the `I` tags after it.

    Args:
        tags (List[str]): `B`, `I` or `O` for each word.

    Returns:
        List[Tuple[int, int]]: The index of the first word of each term and of the word
        after its last.
    """
    spans = []
    j = 0
    while j < len(tags):
        if tags[j] == "B":
            k = j + 1
            while k < len(tags) and tags[k] == "I":
                k += 1
            spans.append((j, k))
            j = k
        else:
            j += 1
    return spans


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
        lemma_layer: str | None = None,
        random_state: int | None = None,
        patience: int | None = None,
        terms_layer: str = GOLD_LAYER,
        sentences_layer: str | None = None,
    ):
        """
        Initialize the Token Classification Term Extractor.

        Args:
            model_name (str): Hugging Face model to use for token classification, or
                the directory of a model saved with `save`, which is already trained.
            n (int): Maximum n-gram size to consider.
            batch_size (int): Batch size for training and evaluation.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for training.
            max_length (int): Maximum sequence length for input tokenization.
            language (str): Language code for tokenization (en, fr, nl).
            stopwords (Union[str, Iterable[str], None]): Drop extracted terms that start or
                end with a stopword: a name in `STOPWORD_LISTS`, a list of words, or None.
            words_layer (Optional[str]): For Teanga input, the span layer holding the
                words, or None for `words` or, failing that, `tokens`.
            lemma_layer (Optional[str]): For Teanga input, a sequence layer, based on
                the words layer, holding the lemma of each word. The model still tags
                the words as written, but each tagged term is given by the lemmas of its
                words, so inflected forms are one term, and labels are matched to the
                lemmas of the words. None gives terms as written.
            random_state (Optional[int]): Seed for the new classification layer of the
                model and the order of the training examples, for reproducible
                training, or None for an unseeded classification layer.
            patience (Optional[int]): With a validation set, stop training when the F1
                on it has not improved for this many epochs. None trains for `epochs`.
            terms_layer (str): For Teanga input without labels, the span layer of gold
                terms to train on.
            sentences_layer (Optional[str]): For Teanga input, the span layer of the
                sentences, or None for `sentences` or, if the corpus has no such layer,
                splitting after each word that is ".", "!" or "?".
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
        self.lemma_layer = lemma_layer
        self.random_state = random_state
        self.patience = patience
        self.terms_layer = terms_layer
        self.sentences_layer = sentences_layer

        # Initialize tokenizers (for preprocessing)
        self.moses_tokenizer = MosesTokenizer(lang=language)
        self.moses_detokenizer = MosesDetokenizer(lang=language)

        # Outside a term, beginning of a term, inside a term
        self.label_list = ["O", "B", "I"]
        self.label_to_id = {label: i for i, label in enumerate(self.label_list)}
        self.id_to_label = dict(enumerate(self.label_list))

        if random_state is not None:
            set_seed(random_state)
        # A model saved by `save` has these labels, and is already trained
        self.is_trained = (
            AutoConfig.from_pretrained(model_name).label2id == self.label_to_id
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(self.label_list),
            id2label=self.id_to_label,
            label2id=self.label_to_id,
        )
        self.trainer: Trainer | None = None

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
        self,
        sentence_list: list[tuple[list[str], str]],
        terms: dict[str, int],
        lemmas: list[list[str]] | None = None,
    ) -> list[tuple[list[str], list[str]]]:
        """
        Create training data with token-level annotations.

        Where labelled terms overlap, the longest is tagged, and of two of the same
        length the first.

        Args:
            sentence_list (List[Tuple[List[str], str]]): List of (tokens, original sentence) tuples.
            terms (Dict[str, int]): Dictionary with terms as keys and binary labels as values.
            lemmas (Optional[List[List[str]]]): The lemma of each token of each
                sentence, to match the terms to the lemmas rather than the tokens, or
                None.

        Returns:
            List[Tuple[List[str], List[str]]]: List of (tokens, tags) tuples.
        """
        training_data = []
        # Only terms labelled 1 are tagged; labels are matched case-insensitively
        positive_terms = {term.lower() for term, label in terms.items() if label == 1}

        for i, (sentence_tokens, original_sentence) in enumerate(sentence_list):
            if not sentence_tokens:
                continue

            tags = ["O"] * len(sentence_tokens)
            for length in range(min(self.n, len(sentence_tokens)), 0, -1):
                for start in range(len(sentence_tokens) - length + 1):
                    end = start + length
                    if any(tag != "O" for tag in tags[start:end]):
                        continue
                    n_gram = (
                        self.moses_detokenizer.detokenize(sentence_tokens[start:end])
                        if lemmas is None
                        else " ".join(lemmas[i][start:end])
                    )
                    if n_gram.lower() in positive_terms:
                        tags[start:end] = ["B"] + ["I"] * (length - 1)

            training_data.append((sentence_tokens, tags))

        return training_data

    def create_span_training_data(
        self,
        sentences: list[tuple[list[str], list[tuple[int, int]]]],
        terms: list[tuple[int, int]],
    ) -> list[tuple[list[str], list[str]]]:
        """
        Create training data from the gold term spans of a document.

        Each term is tagged on the words within its span. Where terms are nested or
        overlap, the longest is tagged, as in ACTER's sequential annotations, and of two
        of the same length the first.

        Args:
            sentences (List[Tuple[List[str], List[Tuple[int, int]]]]): The words of
                each sentence of the document, and their character offsets.
            terms (List[Tuple[int, int]]): The character offsets of the gold terms.

        Returns:
            List[Tuple[List[str], List[str]]]: List of (tokens, tags) tuples, one per
            sentence.
        """
        spans = [span for _, sentence_spans in sentences for span in sentence_spans]
        starts = [start for start, _ in spans]
        covered = []
        for start, end in terms:
            first = last = bisect_left(starts, start)
            while last < len(spans) and spans[last][1] <= end:
                last += 1
            if last > first:
                covered.append((first, last))
        tags = ["O"] * len(spans)
        for first, last in sorted(covered, key=lambda span: (span[0] - span[1], span)):
            if all(tag == "O" for tag in tags[first:last]):
                tags[first:last] = ["B"] + ["I"] * (last - first - 1)

        training_data = []
        offset = 0
        for tokens, _ in sentences:
            sentence_tags = tags[offset : offset + len(tokens)]
            offset += len(tokens)
            # A term that runs over the end of a sentence starts again in the next
            if sentence_tags and sentence_tags[0] == "I":
                sentence_tags[0] = "B"
            training_data.append((tokens, sentence_tags))
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
        self, tags: list[str], tokens: list[str], lemmas: list[str] | None = None
    ) -> list[tuple[str, int, int]]:
        """
        Find the terms tagged in a sentence: each `B` with the `I` tags after it.

        Args:
            tags (List[str]): The tag of each token.
            tokens (List[str]): The tokens of the sentence.
            lemmas (Optional[List[str]]): The lemma of each token, to give each term
                by its lemmas, or None to give it as written.

        Returns:
            List[Tuple[str, int, int]]: Each term in lower case, with the index of its
            first token and of the token after its last.
        """
        return [
            (
                (
                    self.moses_detokenizer.detokenize(tokens[j:k])
                    if lemmas is None
                    else " ".join(lemmas[j:k])
                ).lower(),
                j,
                k,
            )
            for j, k in tag_spans(tags)
        ]

    def train_model(
        self,
        tokenized_dataset: torch.utils.data.Dataset,
        validation_dataset: torch.utils.data.Dataset | None = None,
    ):
        """
        Train the token classification model.

        Args:
            tokenized_dataset: Training dataset.
            validation_dataset: Dataset to compute the F1 of the tagged terms on after
                each epoch, keeping the model of the best epoch, or None to keep the
                model of the last.
        """
        validate = validation_dataset is not None
        if self.patience is not None and not validate:
            raise ValueError("patience needs a validation set to stop training early.")
        # The trained model is kept in memory, so save no logs, and give the Trainer a
        # temporary directory rather than one in the working directory. The best model
        # on the validation set is saved there until training ends.
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
                seed=42 if self.random_state is None else self.random_state,
                save_strategy="epoch" if validate else "no",
                eval_strategy="epoch" if validate else "no",
                save_only_model=True,
                save_total_limit=1,
                load_best_model_at_end=validate,
                metric_for_best_model="f1" if validate else None,
                report_to="none",
            )
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset,
                eval_dataset=validation_dataset,
                compute_metrics=self._f1 if validate else None,
                callbacks=(
                    [EarlyStoppingCallback(self.patience)]
                    if self.patience is not None
                    else None
                ),
            )
            self.trainer.train()
        self.is_trained = True

    def _f1(self, prediction: EvalPrediction) -> dict[str, float]:
        """The F1 of the tagged terms of a validation set, by exact match of their
        words."""
        found = expected = correct = 0
        for scores, labels in zip(prediction.predictions, prediction.label_ids):
            words = labels != -100
            predicted = tag_spans(
                [self.id_to_label[i] for i in np.argmax(scores, axis=-1)[words]]
            )
            gold = tag_spans([self.id_to_label[i] for i in labels[words]])
            found += len(predicted)
            expected += len(gold)
            correct += len(set(predicted) & set(gold))
        return {"f1": 2 * correct / (found + expected) if found + expected else 0.0}

    def save(self, directory: str | Path) -> None:
        """
        Save the trained model and its tokenizer, to pass the directory as `model_name`
        to extract terms without training again.

        Args:
            directory (Union[str, Path]): The directory to save to.

        Raises:
            RuntimeError: If the model has not been trained.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before it is saved.")
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)

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
        Split each document of a Teanga corpus into sentences, by the sentences layer
        or, if there is none, after each word that is ".", "!" or "?".

        Returns:
            For each document, the words and the word spans of each sentence.
        """
        layer = self.sentences_layer
        if layer is not None or SENTENCES_LAYER in corpus.meta:
            layer = find_layer(
                corpus.meta, layer, (SENTENCES_LAYER,), "sentences_layer"
            )
        documents = []
        for doc_id in corpus.doc_ids:
            doc = corpus.doc_by_id(doc_id)
            ends = iter(sorted(end for _, end in doc[layer])) if layer else iter(())
            sentence_end = next(ends, math.inf)
            sentences = []
            tokens: list[str] = []
            spans: list[tuple[int, int]] = []
            for start, end in word_spans(doc, self.words_layer):
                # A word that starts after the end of the sentence starts a new one
                while start >= sentence_end:
                    sentence_end = next(ends, math.inf)
                    if tokens:
                        sentences.append((tokens, spans))
                        tokens, spans = [], []
                tokens.append(doc.text[start:end])
                spans.append((start, end))
                if layer is None and tokens[-1] in (".", "!", "?"):
                    sentences.append((tokens, spans))
                    tokens, spans = [], []
            if tokens:
                sentences.append((tokens, spans))
            documents.append(sentences)
        return documents

    def _teanga_lemmas(
        self,
        corpus: Corpus,
        documents: list[list[tuple[list[str], list[tuple[int, int]]]]],
    ) -> list[list[list[str]]] | None:
        """
        Get the lemma of each word of each sentence from `_teanga_sentences`, or None
        if `lemma_layer` is not set.
        """
        if self.lemma_layer is None:
            return None
        lemmas = []
        for doc_id, sentences in zip(corpus.doc_ids, documents):
            words = iter(
                word_annotations(
                    corpus.doc_by_id(doc_id),
                    self.lemma_layer,
                    (),
                    "lemma_layer",
                    self.words_layer,
                )
            )
            # The sentences hold every word of the document, in order
            lemmas.append([[next(words) for _ in tokens] for tokens, _ in sentences])
        return lemmas

    def _teanga_training_data(
        self, corpus: Corpus, labels: dict[str, int] | None
    ) -> list[tuple[list[str], list[str]]]:
        """
        Tag the words of a Teanga corpus with the terms labelled 1 or, if `labels` is
        None, with the gold terms of `terms_layer`.
        """
        documents = self._teanga_sentences(corpus)
        if labels is not None:
            lemmas = self._teanga_lemmas(corpus, documents)
            return self.create_training_data(
                [(tokens, " ".join(tokens)) for doc in documents for tokens, _ in doc],
                labels,
                None if lemmas is None else [s for doc in lemmas for s in doc],
            )
        layer = find_layer(corpus.meta, self.terms_layer, (), "terms_layer")
        return [
            sentence
            for doc_id, sentences in zip(corpus.doc_ids, documents)
            for sentence in self.create_span_training_data(
                sentences,
                [(span[0], span[1]) for span in corpus.doc_by_id(doc_id)[layer]],
            )
        ]

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
        lemmas: list[list[list[str]]] | None = None,
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
            lemmas (Optional[List[List[List[str]]]]): The lemma of each token of each
                sentence of each document, to give the tagged terms by their lemmas, or
                None to give them as written.

        Returns:
            ATEResults: Results object with extracted terms, and their offsets if
            `spans` is given.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")

        # Tag every sentence, with dummy labels (all "O") for the tokenizer
        texts = [tokens for sentences in documents for tokens, _ in sentences]
        tags = [["O"] * len(tokens) for tokens in texts]
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
                for term, start, end in self.tagged_terms(
                    next(sentence_tags),
                    tokens,
                    None if lemmas is None else lemmas[k][s],
                )
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

    def fit_strings(
        self,
        corpus: list[str],
        labels: dict[str, int],
        validation: list[str] | None = None,
    ) -> None:
        """
        Fine-tune the model on a list of strings, tagging the terms labelled 1.

        Args:
            corpus (List[str]): List of text documents.
            labels (Dict[str, int]): Dictionary with terms as keys and binary labels as values.
            validation (Optional[List[str]]): Documents tagged with the same labels, to
                keep the model of the epoch with the best F1 on them, or None.
        """

        def training_data(documents: list[str]) -> list[tuple[list[str], list[str]]]:
            return self.create_training_data(
                [s for doc in documents for s in self.preprocess_text(doc)], labels
            )

        self._fit(
            training_data(corpus),
            None if validation is None else training_data(validation),
        )

    def fit_teanga(
        self,
        corpus: Corpus,
        labels: dict[str, int] | None = None,
        validation: Corpus | None = None,
    ) -> None:
        """
        Fine-tune the model on a Teanga corpus, tagging its gold terms or the terms
        labelled 1.

        Args:
            corpus: Teanga corpus object.
            labels (Optional[Dict[str, int]]): Dictionary with terms as keys and binary
                labels as values, or None to tag the gold terms of `terms_layer`.
            validation (Optional[Corpus]): A corpus tagged in the same way, to keep the
                model of the epoch with the best F1 on it, or None.
        """
        self._fit(
            self._teanga_training_data(corpus, labels),
            None
            if validation is None
            else self._teanga_training_data(validation, labels),
        )

    def _fit(
        self,
        training_data: list[tuple[list[str], list[str]]],
        validation_data: list[tuple[list[str], list[str]]] | None = None,
    ) -> None:
        def dataset(data: list[tuple[list[str], list[str]]]):
            tokenized = self.tokenize_and_align_labels(
                [tokens for tokens, _ in data], [tags for _, tags in data]
            )
            return self.create_torch_dataset(tokenized)

        self.train_model(
            dataset(training_data),
            None if validation_data is None else dataset(validation_data),
        )

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
            self._teanga_lemmas(corpus, sentences),
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
