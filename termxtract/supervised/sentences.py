import math

from teanga import Corpus

from ..datasets.base import SENTENCES_LAYER
from ..layers import find_layer, word_annotations, word_spans

Sentence = tuple[list[str], list[tuple[int, int]]]
"""The words of a sentence and their character offsets."""


def teanga_sentences(
    corpus: Corpus,
    words_layer: str | None = None,
    sentences_layer: str | None = None,
) -> list[list[Sentence]]:
    """
    Split each document of a Teanga corpus into sentences, by the sentences layer or,
    if there is none, after each word that is ".", "!" or "?".

    Args:
        corpus (Corpus): A Teanga corpus.
        words_layer (Optional[str]): The span layer holding the words, or None for
            `words` or, failing that, `tokens`.
        sentences_layer (Optional[str]): The span layer of the sentences, or None for
            `sentences` or, if the corpus has no such layer, splitting at the words.

    Returns:
        List[List[Sentence]]: For each document, the words and the word spans of each
        sentence.
    """
    layer = sentences_layer
    if layer is not None or SENTENCES_LAYER in corpus.meta:
        layer = find_layer(corpus.meta, layer, (SENTENCES_LAYER,), "sentences_layer")
    documents = []
    for doc_id in corpus.doc_ids:
        doc = corpus.doc_by_id(doc_id)
        ends = iter(sorted(end for _, end in doc[layer])) if layer else iter(())
        sentence_end = next(ends, math.inf)
        sentences = []
        tokens: list[str] = []
        spans: list[tuple[int, int]] = []
        for start, end in word_spans(doc, words_layer):
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


def teanga_lemmas(
    corpus: Corpus,
    documents: list[list[Sentence]],
    lemma_layer: str | None,
    words_layer: str | None = None,
) -> list[list[list[str]]] | None:
    """
    Get the lemma of each word of each sentence from `teanga_sentences`.

    Args:
        corpus (Corpus): The Teanga corpus.
        documents (List[List[Sentence]]): Its sentences, from `teanga_sentences`.
        lemma_layer (Optional[str]): The sequence layer of the lemmas, or None.
        words_layer (Optional[str]): The words layer, as for `teanga_sentences`.

    Returns:
        Optional[List[List[List[str]]]]: The lemmas of each sentence of each document,
        or None if `lemma_layer` is None.
    """
    if lemma_layer is None:
        return None
    lemmas = []
    for doc_id, sentences in zip(corpus.doc_ids, documents):
        words = iter(
            word_annotations(
                corpus.doc_by_id(doc_id), lemma_layer, (), "lemma_layer", words_layer
            )
        )
        # The sentences hold every word of the document, in order
        lemmas.append([[next(words) for _ in tokens] for tokens, _ in sentences])
    return lemmas
