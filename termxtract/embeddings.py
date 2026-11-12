import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
"""The default sentence encoder: a small English model."""


class Encoder:
    """
    A sentence encoder: a Hugging Face model whose token embeddings are averaged over
    each text, as in Sentence-BERT models such as `all-MiniLM-L6-v2`.

    Texts are given as lists of words. Embeddings are normalised to unit length, so the
    dot product of two embeddings is their cosine similarity. The embeddings of texts
    without masked words are cached, so a text is only encoded once.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 32,
        max_length: int = 256,
        device: str | None = None,
    ):
        """
        Load the encoder.

        Args:
            model_name (str): A Hugging Face model with a fast tokenizer that has a
                mask token.
            batch_size (int): The number of texts encoded at once.
            max_length (int): The maximum number of tokens of a text, including the
                special tokens. Longer texts are truncated; use `chunks` to split them.
            device (Optional[str]): The torch device, or None for CUDA if it is
                available and the CPU otherwise.

        Raises:
            ImportError: If the optional 'transformers' dependencies are missing.
        """
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Sentence embeddings require the optional 'transformers' "
                "dependencies. Install them with: pip install 'termxtract[transformers]'"
            ) from e
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.cache: dict[tuple[str, ...], np.ndarray] = {}

    def chunks(self, words: list[str]) -> list[tuple[int, int]]:
        """
        Split a text into chunks that fit in `max_length` tokens, between words.

        Args:
            words (List[str]): The words of the text.

        Returns:
            List[Tuple[int, int]]: The index of the first word of each chunk and of the
            word after its last. A word too long to fit is a chunk of its own, and is
            truncated when it is encoded.
        """
        limit = self.max_length - self.tokenizer.num_special_tokens_to_add()
        lengths = [
            len(ids)
            for ids in self.tokenizer(words, add_special_tokens=False)["input_ids"]
        ]
        spans = []
        start, size = 0, 0
        for i, length in enumerate(lengths):
            if size + length > limit and i > start:
                spans.append((start, i))
                start, size = i, 0
            size += length
        if start < len(words):
            spans.append((start, len(words)))
        return spans

    def encode(
        self, texts: list[list[str]], masks: list[set[int]] | None = None
    ) -> np.ndarray:
        """
        Embed texts, with some of their words masked.

        Args:
            texts (List[List[str]]): The words of each text.
            masks (Optional[List[Set[int]]]): For each text, the indices of the words
                whose tokens are replaced by the mask token, or None to mask nothing.

        Returns:
            np.ndarray: The unit-length embedding of each text, one per row.
        """
        if not texts:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        if masks is None:
            keys = [tuple(text) for text in texts]
            new = list(dict.fromkeys(key for key in keys if key not in self.cache))
            for start in range(0, len(new), self.batch_size):
                batch = new[start : start + self.batch_size]
                embeddings = self._encode([list(key) for key in batch], None)
                self.cache.update(zip(batch, embeddings))
            return np.array([self.cache[key] for key in keys])
        return np.concatenate(
            [
                self._encode(
                    texts[start : start + self.batch_size],
                    masks[start : start + self.batch_size],
                )
                for start in range(0, len(texts), self.batch_size)
            ]
        )

    def _encode(
        self, texts: list[list[str]], masks: list[set[int]] | None
    ) -> np.ndarray:
        """Embed one batch of texts."""
        import torch

        batch = self.tokenizer(
            texts,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if masks is not None:
            for row, masked in enumerate(masks):
                for column, word in enumerate(batch.word_ids(row)):
                    if word is not None and word in masked:
                        batch["input_ids"][row, column] = self.tokenizer.mask_token_id
        with torch.inference_mode():
            tokens = self.model(**batch.to(self.device)).last_hidden_state
        weights = batch["attention_mask"].unsqueeze(-1).to(tokens.dtype)
        mean = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
        return torch.nn.functional.normalize(mean, dim=-1).float().cpu().numpy()
