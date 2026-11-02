import re

from teanga import Corpus

from ..layers import words_with_offsets


def tokenize_strings(corpus: list[str]) -> list[list[str]]:
    """Split each document into lower-case words."""
    return [re.findall(r"\b\w+\b", doc.lower()) for doc in corpus]


def tokenize_teanga(corpus: Corpus, words_layer: str | None = None) -> list[list[str]]:
    """
    Read each document's lower-case words from a Teanga corpus.

    Args:
        corpus (Corpus): A Teanga corpus.
        words_layer (Optional[str]): The span layer holding the words, or None for
            `words` or, failing that, `tokens`.

    Returns:
        List[List[str]]: The words of each document.
    """
    docs = [corpus.doc_by_id(doc_id) for doc_id in corpus.doc_ids]
    return [
        [word.lower() for _, _, word in words_with_offsets(doc, words_layer)]
        for doc in docs
    ]
