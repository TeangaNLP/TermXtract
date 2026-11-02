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
    doc: Document, words_layer: str | None = None
) -> list[tuple[int, int, str]]:
    """
    Get the words of a Teanga document with their offsets.

    Args:
        doc (Document): A Teanga document.
        words_layer (Optional[str]): The span layer holding the words, or None for
            `words` or, failing that, `tokens`.

    Returns:
        List[Tuple[int, int, str]]: A (start, end, text) tuple for each word.
    """
    text = doc.text
    return [
        (start, end, text[start:end]) for start, end in word_spans(doc, words_layer)
    ]


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
        ValueError: If the tag layer is not based on the words layer.
    """
    words = find_layer(doc.meta, words_layer, WORDS_LAYERS, "words_layer")
    layer = find_layer(doc.meta, pos_layer, POS_LAYERS, "pos_layer")
    base = doc.meta[layer].base
    if base != words:
        raise ValueError(
            f"The part-of-speech layer {layer!r} is based on {base!r}, but the words are "
            f"read from {words!r}. Pass words_layer={base!r}."
        )
    tags = [str(tag) for tag in doc[layer]]
    return tags
