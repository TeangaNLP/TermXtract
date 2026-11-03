"""
The ACL RD-TEC 2.0, abstracts of computational linguistics papers with annotated terms.
"""

import re
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterable
from pathlib import Path

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

CLASSES = ("tech", "tool", "lr", "lr-prod", "model", "measure(ment)", "other")
"""
The classes of terms:

- `tech`: methods and techniques, such as `decision-tree approach`.
- `tool`: tools and systems, such as `GE NLToolset`.
- `lr`: language resources, such as `lexicon`.
- `lr-prod`: specific, named language resources, such as `The Wall Street Journal`.
- `model`: models, such as `tri-gram model`.
- `measure(ment)`: measures and measurements, such as `probabilities`.
- `other`: other terms, such as `words`.
"""

ANNOTATORS = (1, 2)
"""The two annotators. Annotator 1 annotated 189 abstracts, annotator 2 annotated 282,
and 171 abstracts were annotated by both."""

LEMMA_LAYER = "lemma"
"""The sequence layer, based on `tokens`, of the lemma of each token."""

POS_LAYER = "pos"
"""The sequence layer, based on `tokens`, of the Penn Treebank tag of each token."""

UPOS_LAYER = "upos"
"""The sequence layer, based on `tokens`, of the Universal Dependencies tag of each
token, converted from its Penn Treebank tag."""

PTB_TO_UPOS = {
    "CC": "CCONJ",
    "CD": "NUM",
    "DT": "DET",
    "EX": "PRON",
    "FW": "X",
    "IN": "ADP",
    "JJ": "ADJ",
    "JJR": "ADJ",
    "JJS": "ADJ",
    "LS": "X",
    "MD": "AUX",
    "NN": "NOUN",
    "NNS": "NOUN",
    "NNP": "PROPN",
    "NNPS": "PROPN",
    "PDT": "DET",
    "POS": "PART",
    "PRP": "PRON",
    "PRP$": "PRON",
    "RB": "ADV",
    "RBR": "ADV",
    "RBS": "ADV",
    "RP": "ADP",
    "SYM": "SYM",
    "TO": "PART",
    "UH": "INTJ",
    "VB": "VERB",
    "VBD": "VERB",
    "VBG": "VERB",
    "VBN": "VERB",
    "VBP": "VERB",
    "VBZ": "VERB",
    "WDT": "DET",
    "WP": "PRON",
    "WP$": "PRON",
    "WRB": "ADV",
    "$": "SYM",
    "#": "SYM",
}
"""The Universal Dependencies tag of each Penn Treebank tag. Other tags, which are all
punctuation, map to `PUNCT`."""

_UPOS_TAGS = sorted({*PTB_TO_UPOS.values(), "PUNCT"})


def load_acl_rd_tec(
    path: str | Path,
    *,
    split: str | None = None,
    annotator: int | None = None,
    labels: Iterable[str] | None = None,
) -> Corpus:
    """
    Load the ACL RD-TEC 2.0 from its LINDAT/CLARIN distribution.

    Each annotated abstract is a document. The titles of the papers are not included,
    since terms are only annotated in the abstracts. One abstract, M92-1025, has no
    sentence markup and no tokens in the distribution, and is left out, so the corpus
    has at most 299 abstracts. The layers are:

    - `text`, the sentences of the abstract separated by spaces, and `source`, its ACL
      Anthology ID (such as `P99-1080`).
    - `tokens`, `sentences`, `lemma` and `pos`, from the distribution's
      part-of-speech-tagged files (tagged with Stanford CoreNLP), and `upos`, their
      tags converted to Universal Dependencies with `PTB_TO_UPOS`, as the
      part-of-speech patterns of the extractors expect.
    - `terms` (`GOLD_LAYER`): the annotated terms, which do not nest.
    - `categories` (`CATEGORY_LAYER`): the class of each term, one of `CLASSES`.

    Args:
        path (Union[str, Path]): The distribution archive.
        split (Optional[str]): Not used: the dataset has no splits.
        annotator (Optional[int]): 1 or 2 to load the abstracts annotated by that
            annotator, with their annotations. None (the default) loads every abstract
            once, with the annotations of annotator 1 where both annotated it, since
            annotator 1 was more consistent in the authors' self-agreement study.
        labels (Optional[Iterable[str]]): The classes of the gold terms, from
            `CLASSES`, or None for all of them.

    Returns:
        Corpus: The abstracts, in order of their ACL Anthology IDs.

    Raises:
        ValueError: If `annotator` or a label is unknown.
    """
    if annotator is not None and annotator not in ANNOTATORS:
        raise ValueError(f"Unknown ACL RD-TEC annotator: {annotator!r}. Use 1 or 2.")
    keep = set(
        CLASSES if labels is None else [labels] if isinstance(labels, str) else labels
    )
    if keep - set(CLASSES):
        raise ValueError(
            f"Unknown ACL RD-TEC labels: {', '.join(sorted(keep - set(CLASSES)))}. "
            f"Use {', '.join(CLASSES)}."
        )

    corpus = Corpus()
    corpus.add_layer_meta("text")
    corpus.add_layer_meta(SOURCE_LAYER)
    corpus.add_layer_meta("tokens", layer_type="span", base="text")
    corpus.add_layer_meta(SENTENCES_LAYER, layer_type="span", base="text")
    corpus.add_layer_meta(LEMMA_LAYER, layer_type="seq", base="tokens", data="string")
    corpus.add_layer_meta(POS_LAYER, layer_type="seq", base="tokens", data="string")
    corpus.add_layer_meta(UPOS_LAYER, layer_type="seq", base="tokens", data=_UPOS_TAGS)
    corpus.add_layer_meta(GOLD_LAYER, layer_type="span", base="text")
    corpus.add_layer_meta(
        CATEGORY_LAYER, layer_type="seq", base=GOLD_LAYER, data=list(CLASSES)
    )

    with zipfile.ZipFile(path) as archive:
        # The annotation file of each abstract, by annotator
        files: dict[str, dict[int, str]] = {}
        for name in archive.namelist():
            match = re.fullmatch(
                r"distribution/annoitation_files/annotator([12])/\d+/(.+)_abstr\.xml",
                name,
            )
            if match:
                files.setdefault(match.group(2), {})[int(match.group(1))] = name
        tagged = {
            number: _read_vertical(
                archive.read(
                    f"distribution/pos_tagged_vertical_files/"
                    f"annotator{number}-acl_rdtec2.vert"
                ).decode("utf-8")
            )
            for number in ANNOTATORS
        }

        for acl_id in sorted(files):
            by_annotator = files[acl_id]
            number = annotator if annotator is not None else min(by_annotator)
            if number not in by_annotator:
                continue
            paper = ET.fromstring(archive.read(by_annotator[number]))
            sentences = [_sentence(s) for s in paper.iter("S")]
            words = tagged[number].get(acl_id, [])
            if not sentences or not words:
                continue

            text = " ".join(sentence for sentence, _ in sentences)
            terms = []
            sentence_spans = []
            offset = 0
            for sentence, spans in sentences:
                sentence_spans.append((offset, offset + len(sentence)))
                terms += [
                    (offset + start, offset + end, label)
                    for start, end, label in spans
                    if label in keep
                ]
                offset += len(sentence) + 1

            doc = corpus.add_doc(text=text, source=acl_id)
            doc.tokens = _align(text, [word for word, _, _ in words], acl_id)
            doc[SENTENCES_LAYER] = sentence_spans
            doc[LEMMA_LAYER] = [lemma for _, lemma, _ in words]
            doc[POS_LAYER] = [tag for _, _, tag in words]
            doc[UPOS_LAYER] = [PTB_TO_UPOS.get(tag, "PUNCT") for _, _, tag in words]
            doc[GOLD_LAYER] = [(start, end) for start, end, _ in terms]
            doc[CATEGORY_LAYER] = [label for _, _, label in terms]
    return corpus


def _sentence(element: ET.Element) -> tuple[str, list[tuple[int, int, str]]]:
    """
    The text of a sentence element, stripped, and the offsets and class of each term in
    it, trimmed of whitespace.
    """
    parts: list[str] = []
    spans = []

    def walk(node: ET.Element) -> None:
        parts.append(node.text or "")
        for child in node:
            start = sum(map(len, parts))
            walk(child)
            if child.tag == "term":
                spans.append((start, sum(map(len, parts)), child.get("class", "")))
            parts.append(child.tail or "")

    walk(element)
    raw = "".join(parts)
    lead = len(raw) - len(raw.lstrip())
    text = raw.strip()
    trimmed = []
    for start, end, label in spans:
        term = raw[start:end]
        start += len(term) - len(term.lstrip()) - lead
        end -= len(term) - len(term.rstrip()) + lead
        if start < end:
            trimmed.append((start, end, label))
    return text, trimmed


def _read_vertical(content: str) -> dict[str, list[tuple[str, str, str]]]:
    """The (word, lemma, tag) triples of each abstract in a vertical file."""
    abstracts: dict[str, list[tuple[str, str, str]]] = {}
    words: list[tuple[str, str, str]] = []
    for line in content.splitlines():
        match = re.match(r'<doc id="([^"]+)"', line)
        if match:
            words = abstracts.setdefault(match.group(1), [])
        elif line.strip() and not line.startswith("<"):
            word, lemma, tag = line.strip().split("\t")
            words.append((word, lemma, tag))
    return abstracts


def _align(text: str, words: list[str], acl_id: str) -> list[tuple[int, int]]:
    """Find the character offsets of the tagged words in the text."""
    spans = []
    pos = 0
    for word in words:
        start = text.find(word, pos)
        if start < 0 or text[pos:start].strip():
            raise ValueError(
                f"The tagged word {word!r} does not match the text of {acl_id} at "
                f"offset {pos}."
            )
        pos = start + len(word)
        spans.append((start, pos))
    return spans


ACL_RD_TEC = register(
    DatasetInfo(
        name="acl-rd-tec",
        description=(
            "ACL RD-TEC 2.0: 300 abstracts of computational linguistics papers, with "
            "terms annotated in context and classified into seven classes."
        ),
        languages=("en",),
        domains=("computational linguistics",),
        license="CC BY-NC-SA 4.0",
        citation=(
            "QasemiZadeh, B., & Schumann, A.-K. (2016). The ACL RD-TEC 2.0: A Language "
            "Resource for Evaluating Term Extraction and Entity Recognition Methods. In "
            "Proceedings of the Tenth International Conference on Language Resources "
            "and Evaluation (LREC 2016)."
        ),
        homepage="http://hdl.handle.net/11372/LRT-1661",
        resources=(
            Resource(
                "https://lindat.mff.cuni.cz/repository/server/api/core/bitstreams/"
                "3d8ee77a-9367-4e10-b512-3d5abc197ce2/content",
                "8c46208c8a10fc82ddcd433deb128834fb44b50384e405a5b5076403f536fcbf",
                "distribution.zip",
            ),
        ),
        loader=load_acl_rd_tec,
    )
)
