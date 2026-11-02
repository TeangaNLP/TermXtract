import re

from teanga import Corpus


def tokenize_strings(corpus: list[str]) -> list[list[str]]:
    """Split each document into lower-case words."""
    return [re.findall(r"\b\w+\b", doc.lower()) for doc in corpus]


def tokenize_teanga(corpus: Corpus) -> list[list[str]]:
    """Read each document's lower-case words from the Teanga `words` layer."""
    docs = [corpus.doc_by_id(doc_id) for doc_id in corpus.doc_ids]
    return [[doc.text[start:end].lower() for start, end in doc.words] for doc in docs]
