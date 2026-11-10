"""
HTEC 2.0, the Hindi Term Extraction Corpus: Hindi Wikipedia articles on education,
with terms annotated in seven categories.
"""

import re
from collections.abc import Iterable
from pathlib import Path

import teanga
from teanga import Corpus

from .base import (
    CATEGORY_LAYER,
    GOLD_LAYER,
    SENTENCES_LAYER,
    SOURCE_LAYER,
    DatasetInfo,
    Resource,
    register,
)

CATEGORIES = (
    "Ambiguous",
    "Educational Institutions, Governing bodies, Think Tanks and Research Institutes",
    "Degrees, Disciplines and different stages of education",
    "Educationists, Learners and Researchers",
    "Education Related Policy and Regulatory Frameworks",
    "Mode of Dissemination",
    "Education Technology and Equipment",
)
"""The categories of the terms, as named in Table 6.1 of Banerjee (2025)."""

# A word is a run of letters, digits and Devanagari vowel signs and other marks, which
# Python does not count as word characters; any other character but a space is a token
_TOKEN = re.compile(r"(?:\w|[ऀ-ःऺ-ॏ॑-ॗॢॣ])+|\S")

# A sentence ends at a line break or after a danda, a double danda, "?", "!" or "."
_SENTENCE = re.compile(r"[^\n]*?(?:[।॥?!.](?=\s|$)|(?=\n)|$)")


def load_htec(
    path: str | Path,
    *,
    split: str | None = None,
    partial_words: bool = False,
    labels: Iterable[str] | None = None,
) -> Corpus:
    """
    Load HTEC 2.0 from its Teanga file.

    Each of the 97 documents is a segment of one of 65 Hindi Wikipedia articles on
    education, with these layers:

    - `text`, and `source`, the document ID (such as `99_a`). Documents with the same
      number come from the same article.
    - `tokens` and `sentences`. The corpus is not tokenised, so the tokens are runs of
      letters, digits and Devanagari signs, with every other character but a space
      a token of its own, and sentences end at line breaks and after `।`, `॥`, `?`,
      `!` and `.`.
    - `terms` (`GOLD_LAYER`): the annotated terms, including nested and overlapping
      ones. Some annotations start inside a word, such as `शोध` in `संशोधित`; by
      default they are left out. Annotations that end inside a word, such as `कक्षा`
      in `कक्षाओं`, are mostly the stems of inflected words, and are kept.
    - `categories` (`CATEGORY_LAYER`): the category of each term, one of
      `CATEGORIES`.

    Args:
        path (Union[str, Path]): The Teanga YAML file of the corpus.
        split (Optional[str]): Not used: the corpus has no splits.
        partial_words (bool): Keep the annotations that start inside a word.
        labels (Optional[Iterable[str]]): The categories of the gold terms, from
            `CATEGORIES`, or None for all of them.

    Returns:
        Corpus: The documents, in order of document ID.

    Raises:
        ValueError: If a label is unknown.
    """
    keep = set(
        CATEGORIES
        if labels is None
        else [labels]
        if isinstance(labels, str)
        else labels
    )
    if keep - set(CATEGORIES):
        raise ValueError(
            f"Unknown HTEC labels: {', '.join(sorted(keep - set(CATEGORIES)))}. Use "
            "the names in termxtract.datasets.htec.CATEGORIES."
        )

    source = teanga.read_yaml(str(path))
    corpus = Corpus()
    corpus.add_layer_meta("text")
    corpus.add_layer_meta(SOURCE_LAYER)
    corpus.add_layer_meta("tokens", layer_type="span", base="text")
    corpus.add_layer_meta(SENTENCES_LAYER, layer_type="span", base="text")
    corpus.add_layer_meta(GOLD_LAYER, layer_type="span", base="text")
    corpus.add_layer_meta(
        CATEGORY_LAYER, layer_type="seq", base=GOLD_LAYER, data=list(CATEGORIES)
    )
    for doc_id in source.doc_ids:
        original = source.doc_by_id(doc_id)
        text = original.text
        tokens = [match.span() for match in _TOKEN.finditer(text)]
        starts = {start for start, _ in tokens}
        terms = [
            (span[0], span[1], category)
            for span, category in zip(original["terms"], original["categories"])
            if category in keep and (partial_words or span[0] in starts)
        ]
        doc = corpus.add_doc(text=text, source=original["source"].text[0])
        doc.tokens = tokens
        doc[SENTENCES_LAYER] = _sentences(text)
        doc[GOLD_LAYER] = [(start, end) for start, end, _ in terms]
        doc[CATEGORY_LAYER] = [category for _, _, category in terms]
    return corpus


def _sentences(text: str) -> list[tuple[int, int]]:
    """The spans of the sentences of a text, without the spaces around them."""
    spans = []
    for match in _SENTENCE.finditer(text):
        sentence = match.group()
        start = match.start() + len(sentence) - len(sentence.lstrip())
        end = match.start() + len(sentence.rstrip())
        if start < end:
            spans.append((start, end))
    return spans


HTEC = register(
    DatasetInfo(
        name="htec",
        description=(
            "HTEC 2.0: 97 segments of Hindi Wikipedia articles on education, with "
            "terms in seven categories."
        ),
        languages=("hi",),
        domains=("education",),
        license="CC BY-SA 4.0, as the Hindi Wikipedia texts",
        citation=(
            "Banerjee, S., Chakravarthi, B. R., & McCrae, J. P. (2025). Benchmarking "
            "Hindi term extraction in education: A dataset and analysis. In "
            "Proceedings of Language, Data and Knowledge (LDK 2025)."
        ),
        homepage="https://zenodo.org/records/TODO",
        resources=(
            Resource(
                "https://zenodo.org/records/TODO/files/htec-2.0.yaml",
                "db276582ffee1b27aee084d11b36f9321748e0161b3d0c06e2e84bcc1494d62c",
                "htec-2.0.yaml",
            ),
        ),
        loader=load_htec,
    )
)
