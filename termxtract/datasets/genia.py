"""
The GENIA term corpus 3.02: MEDLINE abstracts on transcription factors in human blood
cells, with biomedical terms annotated in 36 classes.
"""

import re
import tarfile
import xml.etree.ElementTree as ET
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

CLASSES = (
    "amino_acid_monomer",
    "atom",
    "body_part",
    "carbohydrate",
    "cell_component",
    "cell_line",
    "cell_type",
    "DNA_domain_or_region",
    "DNA_family_or_group",
    "DNA_molecule",
    "DNA_N/A",
    "DNA_substructure",
    "inorganic",
    "lipid",
    "mono_cell",
    "multi_cell",
    "nucleotide",
    "other_artificial_source",
    "other_name",
    "other_organic_compound",
    "peptide",
    "polynucleotide",
    "protein_complex",
    "protein_domain_or_region",
    "protein_family_or_group",
    "protein_molecule",
    "protein_N/A",
    "protein_subunit",
    "protein_substructure",
    "RNA_domain_or_region",
    "RNA_family_or_group",
    "RNA_molecule",
    "RNA_N/A",
    "RNA_substructure",
    "tissue",
    "virus",
)
"""The classes of the GENIA ontology that terms are annotated with."""

_CORPUS = "GENIA_term_3.02/GENIAcorpus3.02.xml"


def load_genia(
    path: str | Path,
    *,
    split: str | None = None,
    nested: bool = False,
    labels: Iterable[str] | None = None,
) -> Corpus:
    """
    Load the GENIA term corpus from its archive.

    Each of the 2,000 MEDLINE abstracts is a document, with these layers:

    - `text`, the title and abstract, their sentences separated by spaces, and
      `source`, the MEDLINE identifier (such as `MEDLINE:95369245`).
    - `sentences`, from the corpus, and `tokens`. The corpus is not tokenised, so the
      tokens are the text split at whitespace, with brackets, quotes and punctuation
      split off the ends of each token unless they close a bracket inside it. Words
      such as `IL-2`, `Ca2+` and `1,25(OH)2D3` stay whole, so that extracted terms
      match the gold terms.
    - `terms` (`GOLD_LAYER`): the annotated terms. By default only the outermost
      terms are kept: in `<IL-2 gene> expression`, annotated as a term with the term
      `IL-2 gene` inside it, only `IL-2 gene expression`. Coordinated terms, such as
      `alpha- and beta-globin`, are left out, since their text is not a term, but
      complete terms inside them are kept.
    - `categories` (`CATEGORY_LAYER`): the class of each term, one of `CLASSES`.
      Where the same text is annotated twice, nested in itself, it is one term with
      the outer class.

    Args:
        path (Union[str, Path]): The corpus archive.
        split (Optional[str]): Not used: the corpus has no splits.
        nested (bool): Keep the terms nested inside other terms too.
        labels (Optional[Iterable[str]]): The classes of the gold terms, from
            `CLASSES`, or None for all of them. With `nested=False`, the outermost of
            the terms with these classes are kept.

    Returns:
        Corpus: The abstracts, in the order of the corpus.

    Raises:
        ValueError: If a label is unknown.
    """
    keep = set(
        CLASSES if labels is None else [labels] if isinstance(labels, str) else labels
    )
    if keep - set(CLASSES):
        raise ValueError(
            f"Unknown GENIA labels: {', '.join(sorted(keep - set(CLASSES)))}. Use "
            "the names in termxtract.datasets.genia.CLASSES."
        )

    corpus = Corpus()
    corpus.add_layer_meta("text")
    corpus.add_layer_meta(SOURCE_LAYER)
    corpus.add_layer_meta("tokens", layer_type="span", base="text")
    corpus.add_layer_meta(SENTENCES_LAYER, layer_type="span", base="text")
    corpus.add_layer_meta(GOLD_LAYER, layer_type="span", base="text")
    corpus.add_layer_meta(
        CATEGORY_LAYER, layer_type="seq", base=GOLD_LAYER, data=list(CLASSES)
    )

    with tarfile.open(path) as archive:
        member = archive.extractfile(_CORPUS)
        if member is None:
            raise ValueError(f"The GENIA archive has no {_CORPUS}.")
        # The DOCTYPE names a DTD that is not loaded; ElementTree ignores it
        root = ET.parse(member).getroot()

    for article in root.iter("article"):
        source = article.findtext("articleinfo/bibliomisc", default="").strip()
        texts = []
        sentences = []
        terms = []
        offset = 0
        for element in article.iter("sentence"):
            sentence, spans = _sentence(element)
            texts.append(sentence)
            sentences.append((offset, offset + len(sentence)))
            terms += [
                (offset + start, offset + end, label)
                for start, end, label in spans
                if label in keep
            ]
            offset += len(sentence) + 1
        text = " ".join(texts)
        terms = _dedupe(terms)
        if not nested:
            terms = _outermost(terms)

        doc = corpus.add_doc(text=text, source=source)
        doc.tokens = _tokenize(text)
        doc[SENTENCES_LAYER] = sentences
        doc[GOLD_LAYER] = [(start, end) for start, end, _ in terms]
        doc[CATEGORY_LAYER] = [label for _, _, label in terms]
    return corpus


def _sentence(element: ET.Element) -> tuple[str, list[tuple[int, int, str]]]:
    """
    The text of a sentence element and the offsets and class of each term in it, from
    the outermost in, leaving out coordinated terms and the parts of them that have no
    class.
    """
    parts: list[str] = []
    spans: list[tuple[int, int, str]] = []

    def walk(node: ET.Element) -> None:
        parts.append(node.text or "")
        for child in node:
            start = sum(map(len, parts))
            index = len(spans)
            walk(child)
            sem = child.get("sem") or ""
            if child.tag == "cons" and sem.startswith("G#"):
                spans.insert(index, (start, sum(map(len, parts)), sem[2:]))
            parts.append(child.tail or "")

    walk(element)
    return "".join(parts), spans


def _dedupe(terms: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """The terms sorted by offset, keeping the first (outer) class of a repeated span."""
    seen: dict[tuple[int, int], str] = {}
    for start, end, label in terms:
        seen.setdefault((start, end), label)
    return sorted(
        ((start, end, label) for (start, end), label in seen.items()),
        key=lambda t: (t[0], -t[1]),
    )


def _outermost(terms: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """The terms not inside another term, from terms sorted by start and then length."""
    kept: list[tuple[int, int, str]] = []
    for term in terms:
        if not kept or term[0] >= kept[-1][1]:
            kept.append(term)
    return kept


_OPENERS = "([{\"'"
_CLOSERS = {")": "(", "]": "[", "}": "{"}
_TRAILING = ",.;:!?\"'"


def _tokenize(text: str) -> list[tuple[int, int]]:
    """
    Split text at whitespace, then split brackets, quotes and punctuation off the ends
    of each token, keeping a closing bracket that closes a bracket inside the token.
    """
    tokens = []
    for match in re.finditer(r"\S+", text):
        start, end = match.span()
        before, after = [], []
        while start < end:
            char = text[end - 1]
            if text[start] in _OPENERS:
                before.append((start, start + 1))
                start += 1
            elif char in _TRAILING or (
                char in _CLOSERS
                and text.count(_CLOSERS[char], start, end - 1)
                <= text.count(char, start, end - 1)
            ):
                after.append((end - 1, end))
                end -= 1
            else:
                break
        tokens += before + ([(start, end)] if start < end else []) + after[::-1]
    return tokens


GENIA = register(
    DatasetInfo(
        name="genia",
        description=(
            "GENIA term corpus 3.02: 2,000 MEDLINE abstracts on transcription factors "
            "in human blood cells, with biomedical terms in 36 classes."
        ),
        languages=("en",),
        domains=("biomedicine",),
        license=(
            "GENIA Project License for Annotated Corpora: annotations CC BY 3.0 "
            '("Corpus annotations (c) GENIA Project"); abstracts under PubMed terms'
        ),
        citation=(
            "Kim, J.-D., Ohta, T., Tateisi, Y., & Tsujii, J. (2003). GENIA corpus - a "
            "semantically annotated corpus for bio-textmining. Bioinformatics, "
            "19(suppl. 1), i180-i182. https://doi.org/10.1093/bioinformatics/btg1023"
        ),
        homepage="http://www.geniaproject.org/genia-corpus/term-corpus",
        resources=(
            Resource(
                "https://www.nactem.ac.uk/GENIA/current/GENIA-corpus/Term/"
                "GENIAcorpus3.02.tgz",
                "85b2aede313308c33783ba508d054d0af0ff9c7e48490f0db5168a9f88d74d1c",
                "GENIAcorpus3.02.tgz",
            ),
        ),
        loader=load_genia,
    )
)
