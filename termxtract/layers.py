from typing import Any

from teanga import Document

WORDS_LAYERS = ("words", "tokens")
"""Layers read as the words of a Teanga document when no `words_layer` is given."""

POS_LAYERS = ("upos", "pos")
"""Layers read as part-of-speech tags when no `pos_layer` is given."""


def find_layer(
    meta: dict[str, Any],
    layer: str | None,
    defaults: tuple[str, ...],
    parameter: str,
) -> str:
    """
    Choose the layer to read from a Teanga corpus.

    Args:
        meta (Dict[str, Any]): The layer metadata of the corpus.
        layer (Optional[str]): The layer requested by the user, or None for the first of
            `defaults` that the corpus has.
        defaults (Tuple[str, ...]): The layers to try, in order, when `layer` is None.
        parameter (str): The name of the parameter that sets the layer, for messages.

    Returns:
        str: The name of the layer.

    Raises:
        ValueError: If the requested layer, or every default layer, is missing.
    """
    if layer is not None:
        if layer not in meta:
            raise ValueError(f"The Teanga corpus has no {layer!r} layer.")
        return layer
    for name in defaults:
        if name in meta:
            return name
    raise ValueError(
        f"The Teanga corpus has none of the layers {', '.join(map(repr, defaults))}. "
        f"Pass {parameter}=... to name the layer to use."
    )


def word_spans(doc: Document, words_layer: str | None = None) -> list[tuple[int, int]]:
    """
    Get the character offsets of the words of a Teanga document.

    Args:
        doc (Document): A Teanga document.
        words_layer (Optional[str]): The span layer holding the words, or None for
            `words` or, failing that, `tokens`.

    Returns:
        List[Tuple[int, int]]: The (start, end) offsets of each word.
    """
    layer = find_layer(doc.meta, words_layer, WORDS_LAYERS, "words_layer")
    return [(span[0], span[1]) for span in doc[layer]]


def words_with_offsets(
    doc: Document, words_layer: str | None = None, lemma_layer: str | None = None
) -> list[tuple[int, int, str]]:
    """
    Get the words of a Teanga document with their offsets.

    Args:
        doc (Document): A Teanga document.
        words_layer (Optional[str]): The span layer holding the words, or None for
            `words` or, failing that, `tokens`.
        lemma_layer (Optional[str]): A sequence layer, based on the words layer, holding
            the lemma of each word. If given, each word is its lemma, so that inflected
            forms are counted together; if None, each word is its text.

    Returns:
        List[Tuple[int, int, str]]: A (start, end, text) tuple for each word, where text
        is the lemma if `lemma_layer` is given.
    """
    spans = word_spans(doc, words_layer)
    if lemma_layer is None:
        text = doc.text
        return [(start, end, text[start:end]) for start, end in spans]
    lemmas = word_annotations(doc, lemma_layer, (), "lemma_layer", words_layer)
    return [(start, end, lemma) for (start, end), lemma in zip(spans, lemmas)]


def word_annotations(
    doc: Document,
    layer: str | None,
    defaults: tuple[str, ...],
    parameter: str,
    words_layer: str | None = None,
) -> list[str]:
    """
    Get the value of a sequence layer, such as tags or lemmas, for each word of a document.

    Args:
        doc (Document): A Teanga document.
        layer (Optional[str]): The sequence layer, or None for the first of `defaults`
            that the corpus has.
        defaults (Tuple[str, ...]): The layers to try when `layer` is None.
        parameter (str): The name of the parameter that sets the layer, for messages.
        words_layer (Optional[str]): The words layer, as for `word_spans`.

    Returns:
        List[str]: One value per word.

    Raises:
        ValueError: If the layer is missing or not based on the words layer.
    """
    words = find_layer(doc.meta, words_layer, WORDS_LAYERS, "words_layer")
    name = find_layer(doc.meta, layer, defaults, parameter)
    base = doc.meta[name].base
    if base != words:
        raise ValueError(
            f"The layer {name!r} is based on {base!r}, but the words are read from "
            f"{words!r}. Pass words_layer={base!r}."
        )
    return [str(value) for value in doc[name]]


def pos_tags(
    doc: Document, pos_layer: str | None = None, words_layer: str | None = None
) -> list[str]:
    """
    Get the part-of-speech tag of each word of a Teanga document.

    Args:
        doc (Document): A Teanga document.
        pos_layer (Optional[str]): The sequence layer holding the tags, or None for
            `upos` or, failing that, `pos`.
        words_layer (Optional[str]): The words layer, as for `word_spans`.

    Returns:
        List[str]: One tag per word.

    Raises:
        ValueError: If the tag layer is missing or not based on the words layer.
    """
    return word_annotations(doc, pos_layer, POS_LAYERS, "pos_layer", words_layer)
