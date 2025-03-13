import re
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from collections import Counter
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    Trainer, 
    TrainingArguments
)
from sacremoses import MosesTokenizer, MosesDetokenizer
from sklearn.model_selection import train_test_split
from teanga import Corpus
from ..utils import ATEResults


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
        language: str = "en"
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
        """
        self.model_name = model_name
        self.n = n
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.language = language
        
        # Initialize tokenizers (for preprocessing)
        self.moses_tokenizer = MosesTokenizer(lang=language)
        self.moses_detokenizer = MosesDetokenizer(lang=language)
        
        # Label mapping for token classification
        self.label_list = ["n", "B-T", "T"]  # non-term, term beginning, term continuation
        self.label_to_id = {l: i for i, l in enumerate(self.label_list)}
        self.id_to_label = {i: l for i, l in enumerate(self.label_list)}
        
        # Initialize transformer model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, 
            num_labels=len(self.label_list)
        )
        
        # Initialize trainer
        self.trainer = None
        self.is_trained = False
    
    def preprocess_text(self, text: str) -> List[Tuple[List[str], str]]:
        """
        Split text into sentences and tokenize.
        
        Args:
            text (str): Input text to preprocess.
            
        Returns:
            List[Tuple[List[str], str]]: List of tuples (tokens, original sentence).
        """
        # Simple sentence splitting by common delimiters
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        sentence_list = []
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Tokenize using Moses tokenizer
            tokenized_text = self.moses_tokenizer.tokenize(sentence, return_str=True)
            sentence_list.append((tokenized_text.split(), sentence))
            
        return sentence_list
    
    def find_sublist_indices(self, sublist: List[str], full_list: List[str]) -> List[Tuple[int, int]]:
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
            if full_list[ind:ind + sublist_len] == sublist:
                results.append((ind, ind + sublist_len - 1))
                
        return results
    
    def create_training_data(
        self, 
        sentence_list: List[Tuple[List[str], str]], 
        terms: Dict[str, int]
    ) -> List[Tuple[List[str], List[str]]]:
        """
        Create training data with token-level annotations.
        
        Args:
            sentence_list (List[Tuple[List[str], str]]): List of (tokens, original sentence) tuples.
            terms (Dict[str, int]): Dictionary with terms as keys and binary labels as values.
            
        Returns:
            List[Tuple[List[str], List[str]]]: List of (tokens, tags) tuples.
        """
        training_data = []
        
        for sentence_tokens, original_sentence in sentence_list:
            if not sentence_tokens:
                continue
                
            # Initialize all tokens as non-terms
            tags = ["n"] * len(sentence_tokens)
            
            # Iterate through n-grams from 1 to n
            for i in range(1, min(self.n + 1, len(sentence_tokens) + 1)):
                # Create n-grams
                for start_idx in range(len(sentence_tokens) - i + 1):
                    n_gram_tokens = sentence_tokens[start_idx:start_idx + i]
                    n_gram_text = self.moses_detokenizer.detokenize(n_gram_tokens)
                    
                    # Check if n-gram is in the term list
                    if n_gram_text.lower() in terms:
                        # Mark the beginning token as B-T
                        tags[start_idx] = "B-T"
                        
                        # Mark continuation tokens as T
                        for j in range(start_idx + 1, start_idx + i):
                            tags[j] = "T"
            
            training_data.append((sentence_tokens, tags))
            
        return training_data
    
    def tokenize_and_align_labels(
        self, 
        texts: List[List[str]], 
        tags: List[List[str]]
    ) -> Dict:
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
            return_tensors="pt"
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
    
    def create_torch_dataset(self, tokenized_inputs: Dict) -> torch.utils.data.Dataset:
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
        self, 
        token_predictions: List[List[str]], 
        texts: List[List[str]]
    ) -> Dict[str, float]:
        """
        Extract terms from token-level predictions.
        
        Args:
            token_predictions (List[List[str]]): Token-level predictions.
            texts (List[List[str]]): Original tokenized texts.
            
        Returns:
            Dict[str, float]: Dictionary of extracted terms with confidence scores.
        """
        extracted_terms = {}
        
        for i, pred in enumerate(token_predictions):
            txt = texts[i]
            j = 0
            
            while j < len(pred):
                # If beginning of a term is found
                if pred[j] == "B-T":
                    term = [txt[j]]
                    k = j + 1
                    
                    # Continue adding tokens as long as they are part of the term
                    while k < len(pred) and pred[k] == "T":
                        term.append(txt[k])
                        k += 1
                    
                    # Add the complete term
                    extracted_term = self.moses_detokenizer.detokenize(term)
                    
                    # For simplicity, set confidence to 1.0
                    # In a real implementation, you would use model confidence scores
                    extracted_terms[extracted_term.lower()] = 1.0
                    
                    # Move to the token after the term
                    j = k
                else:
                    j += 1
                    
        return extracted_terms
    
    def train_model(self, tokenized_dataset):
        """
        Train the token classification model.
        
        Args:
            tokenized_dataset: Training dataset.
        """
        # Initialize training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size * 2,
            warmup_steps=0,
            weight_decay=0.01,
            learning_rate=self.learning_rate,
            logging_dir='./logs',
            logging_steps=100,
            save_steps=500,
            evaluation_strategy="no"
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset
        )
        
        # Train the model
        self.trainer.train()
        self.is_trained = True
    
    def predict(self, tokenized_dataset, texts) -> Dict[str, float]:
        """
        Make predictions using the trained model.
        
        Args:
            tokenized_dataset: Dataset for prediction.
            texts (List[List[str]]): Original tokenized texts.
            
        Returns:
            Dict[str, float]: Dictionary of extracted terms with confidence scores.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")
            
        if not self.trainer:
            self.trainer = Trainer(
                model=self.model,
                args=TrainingArguments(output_dir='./results')
            )
        
        # Get predictions
        predictions, labels, _ = self.trainer.predict(tokenized_dataset)
        predictions = np.argmax(predictions, axis=2)
        
        # Convert to label names
        true_predictions = [
            [self.id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        # Extract terms
        return self.extract_terms_from_predictions(true_predictions, texts)
    
    def prepare_teanga_corpus(self, corpus) -> List[Tuple[List[str], str]]:
        """
        Prepare a Teanga corpus for term extraction.
        
        Args:
            corpus: Teanga corpus object.
            
        Returns:
            List[Tuple[List[str], str]]: List of (tokens, original sentence) tuples.
        """
        sentence_list = []
        
        for doc_id in corpus.doc_ids:
            doc = corpus.doc_by_id(doc_id)
            
            # Extract tokens from document
            tokens = []
            for start, end in doc.words:
                tokens.append(doc.text[start:end])
                
            # Group tokens into sentences (simplistic approach)
            current_sentence = []
            current_text = ""
            
            for token in tokens:
                current_sentence.append(token)
                current_text += token + " "
                
                # If sentence-ending punctuation, add to sentence list
                if token in [".", "!", "?"]:
                    if current_sentence:
                        sentence_list.append((current_sentence, current_text.strip()))
                        current_sentence = []
                        current_text = ""
            
            # Add any remaining tokens as a sentence
            if current_sentence:
                sentence_list.append((current_sentence, current_text.strip()))
                
        return sentence_list
    
    def extract_terms_teanga(self, corpus, labels: Dict[str, int] = None) -> ATEResults:
        """
        Extract terms from a Teanga corpus.
        
        Args:
            corpus: Teanga corpus object.
            labels (Dict[str, int]): Dictionary with terms as keys and binary labels as values (for training).
            
        Returns:
            ATEResults: Results object with extracted terms.
        """
        # If labels are provided, we're in training mode
        if labels is not None:
            return self.train_terms_teanga(corpus, labels)
        
        # Prepare corpus for prediction
        sentence_list = self.prepare_teanga_corpus(corpus)
        
        # Create dummy labels (all "n") for the tokenizer
        training_data = []
        for tokens, _ in sentence_list:
            training_data.append((tokens, ["n"] * len(tokens)))
        
        # Separate tokens and tags
        texts = [tup[0] for tup in training_data]
        tags = [tup[1] for tup in training_data]
        
        # Tokenize for the model
        tokenized_inputs = self.tokenize_and_align_labels(texts, tags)
        tokenized_dataset = self.create_torch_dataset(tokenized_inputs)
        
        # Get predictions
        term_scores = self.predict(tokenized_dataset, texts)
        
        # Create results object
        terms_by_doc = []
        for doc_id in corpus.doc_ids:
            doc_terms = []
            doc_text = corpus.doc_by_id(doc_id).text.lower()
            
            for term, score in term_scores.items():
                # Check if term appears in the document
                if term.lower() in doc_text:
                    doc_terms.append({"term": term, "score": score})
            
            terms_by_doc.append({"doc_id": doc_id, "terms": doc_terms})
        
        return ATEResults(corpus=corpus, terms=terms_by_doc)
    
    def train_terms_teanga(self, corpus, labels: Dict[str, int]) -> ATEResults:
        """
        Train the model and extract terms from a Teanga corpus.
        
        Args:
            corpus: Teanga corpus object.
            labels (Dict[str, int]): Dictionary with terms as keys and binary labels as values.
            
        Returns:
            ATEResults: Results object with extracted terms.
        """
        # Prepare corpus for training
        sentence_list = self.prepare_teanga_corpus(corpus)
        
        # Create training data
        training_data = self.create_training_data(sentence_list, labels)
        
        # Separate tokens and tags
        train_texts = [tup[0] for tup in training_data]
        train_tags = [tup[1] for tup in training_data]
        
        # Tokenize for the model
        train_tokenized = self.tokenize_and_align_labels(train_texts, train_tags)
        
        # Create dataset
        train_dataset = self.create_torch_dataset(train_tokenized)
        
        # Train the model
        self.train_model(train_dataset)
        
        # Extract terms from the corpus
        return self.extract_terms_teanga(corpus)
    
    def extract_terms_strings(self, corpus: List[str], labels: Dict[str, int] = None) -> ATEResults:
        """
        Extract terms from a list of strings.
        
        Args:
            corpus (List[str]): List of text documents.
            labels (Dict[str, int]): Dictionary with terms as keys and binary labels as values (for training).
            
        Returns:
            ATEResults: Results object with extracted terms.
        """
        # If labels are provided, we're in training mode
        if labels is not None:
            return self.train_terms_strings(corpus, labels)
        
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        
        # Preprocess text
        sentence_list = []
        for doc in corpus:
            sentence_list.extend(self.preprocess_text(doc))
        
        # Create dummy labels (all "n") for the tokenizer
        training_data = []
        for tokens, _ in sentence_list:
            training_data.append((tokens, ["n"] * len(tokens)))
        
        # Separate tokens and tags
        texts = [tup[0] for tup in training_data]
        tags = [tup[1] for tup in training_data]
        
        # Tokenize for the model
        tokenized_inputs = self.tokenize_and_align_labels(texts, tags)
        tokenized_dataset = self.create_torch_dataset(tokenized_inputs)
        
        # Get predictions
        term_scores = self.predict(tokenized_dataset, texts)
        
        # Create results object
        terms_by_doc = []
        for i, doc in enumerate(corpus):
            doc_terms = []
            doc_lower = doc.lower()
            
            for term, score in term_scores.items():
                # Check if term appears in the document
                if term.lower() in doc_lower:
                    doc_terms.append({"term": term, "score": score})
            
            terms_by_doc.append({"doc_id": f"doc_{i}", "terms": doc_terms})
        
        return ATEResults(corpus=corpus, terms=terms_by_doc)
    
    def train_terms_strings(self, corpus: List[str], labels: Dict[str, int]) -> ATEResults:
        """
        Train the model and extract terms from a list of strings.
        
        Args:
            corpus (List[str]): List of text documents.
            labels (Dict[str, int]): Dictionary with terms as keys and binary labels as values.
            
        Returns:
            ATEResults: Results object with extracted terms.
        """
        # Preprocess text
        sentence_list = []
        for doc in corpus:
            sentence_list.extend(self.preprocess_text(doc))
        
        # Create training data
        training_data = self.create_training_data(sentence_list, labels)
        
        # Separate tokens and tags
        train_texts = [tup[0] for tup in training_data]
        train_tags = [tup[1] for tup in training_data]
        
        # Tokenize for the model
        train_tokenized = self.tokenize_and_align_labels(train_texts, train_tags)
        
        # Create dataset
        train_dataset = self.create_torch_dataset(train_tokenized)
        
        # Train the model
        self.train_model(train_dataset)
        
        # Extract terms from the corpus
        return self.extract_terms_strings(corpus)