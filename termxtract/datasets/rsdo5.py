"""
RSDO5 1.1: twelve Slovenian texts in four domains, with every term annotated and marked
as in-domain or out-domain.
"""

import re
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

DOMAINS = ("bim", "jez", "kem", "vet")
"""The domains of RSDO5, as in its file names: biomechanics (biomehanika), linguistics
(jezikoslovje), chemistry (kemija) and veterinary science (veterina)."""

LABELS = ("in-domain", "out-domain")
"""The labels of the terms: of the domain of the text, or of another field."""

_ATTRIBUTE = re.compile(r'(\w+)="([^"]*)"')


def load_rsdo5(
    path: str | Path,
    *,
    split: str | None = None,
    domains: str | Iterable[str] | None = None,
    labels: Iterable[str] | None = None,
) -> Corpus:
    """
    Load RSDO5 from the archive of its vertical files.

    Each of the 12 texts (a doctoral thesis, a textbook and one or two journal
    articles per domain) is a document, with these layers:

    - `text`, rebuilt from the tokens, with a line break between paragraphs, and
      `source`, the ID of the text (such as `rsdo5bimdis`).
    - `tokens` and `sentences`, from the corpus's Universal Dependencies annotation.
    - `lemma`, `upos` (Universal Dependencies) and `msd` (MULTEXT-East) layers, based
      on `tokens`.
    - `terms` (`GOLD_LAYER`): the annotated terms. They do not nest.
    - `categories` (`CATEGORY_LAYER`): whether each term is `in-domain` or
      `out-domain`, one of `LABELS`.

    Args:
        path (Union[str, Path]): The archive of the vertical files.
        split (Optional[str]): Not used: the corpus has no splits. To train on some
            domains and test on another, load each with `domains`.
        domains (Optional[Union[str, Iterable[str]]]): Domains to load, from
            `DOMAINS`, or None for all four.
        labels (Optional[Iterable[str]]): The labels of the gold terms, from `LABELS`,
            or None for both.

    Returns:
        Corpus: The texts, in order of ID.

    Raises:
        ValueError: If a domain or label is unknown.
    """
    chosen = set(
        DOMAINS
        if domains is None
        else [domains]
        if isinstance(domains, str)
        else domains
    )
    if chosen - set(DOMAINS):
        raise ValueError(
            f"Unknown RSDO5 domains: {', '.join(sorted(chosen - set(DOMAINS)))}. Use "
            f"{', '.join(DOMAINS)}."
        )
    keep = set(
        LABELS if labels is None else [labels] if isinstance(labels, str) else labels
    )
    if keep - set(LABELS):
        raise ValueError(
            f"Unknown RSDO5 labels: {', '.join(sorted(keep - set(LABELS)))}. Use "
            f"{', '.join(LABELS)}."
        )

    corpus = Corpus()
    corpus.add_layer_meta("text")
    corpus.add_layer_meta(SOURCE_LAYER)
    corpus.add_layer_meta("tokens", layer_type="span", base="text")
    corpus.add_layer_meta(SENTENCES_LAYER, layer_type="span", base="text")
    corpus.add_layer_meta("lemma", layer_type="seq", base="tokens", data="string")
    corpus.add_layer_meta("upos", layer_type="seq", base="tokens", data="string")
    corpus.add_layer_meta("msd", layer_type="seq", base="tokens", data="string")
    corpus.add_layer_meta(GOLD_LAYER, layer_type="span", base="text")
    corpus.add_layer_meta(
        CATEGORY_LAYER, layer_type="seq", base=GOLD_LAYER, data=list(LABELS)
    )

    with zipfile.ZipFile(path) as archive:
        names = sorted(
            name
            for name in archive.namelist()
            if re.fullmatch(r".*/rsdo5(\w{3})\w+\.vert", name)
            and name.rsplit("/", 1)[1][5:8] in chosen
        )
        for name in names:
            _add_text(corpus, archive.read(name).decode("utf-8"), keep)
    return corpus


def _add_text(corpus: Corpus, vertical: str, keep: set[str]) -> None:
    """Add a text in the vertical format to the corpus."""
    parts: list[str] = []
    length = 0
    separator = ""
    source = ""
    tokens: list[tuple[int, int]] = []
    lemmas: list[str] = []
    upos: list[str] = []
    msd: list[str] = []
    sentences: list[tuple[int, int]] = []
    terms: list[tuple[int, int, str]] = []
    sentence_start = term_start = 0
    label = ""
    for line in vertical.splitlines():
        if line.startswith("<text"):
            source = dict(_ATTRIBUTE.findall(line))["id"]
        elif line.startswith("<p"):
            separator = "\n" if parts else ""
        elif line == "<g/>":
            separator = ""
        elif line.startswith("<s"):
            sentence_start = length + len(separator)
        elif line.startswith("</s"):
            sentences.append((sentence_start, length))
        elif line.startswith("<term"):
            term_start = length + len(separator)
            label = dict(_ATTRIBUTE.findall(line))["domain"]
        elif line.startswith("</term"):
            if label in keep:
                terms.append((term_start, length, label))
        elif line and not line.startswith("<"):
            fields = line.split("\t")
            parts += [separator, fields[0]]
            start = length + len(separator)
            length = start + len(fields[0])
            tokens.append((start, length))
            lemmas.append(fields[1])
            msd.append(fields[2])
            upos.append(fields[3].removeprefix("UPosTag="))
            separator = " "
    doc = corpus.add_doc(text="".join(parts), source=source)
    doc.tokens = tokens
    doc[SENTENCES_LAYER] = sentences
    doc.lemma = lemmas
    doc.upos = upos
    doc.msd = msd
    doc[GOLD_LAYER] = [(start, end) for start, end, _ in terms]
    doc[CATEGORY_LAYER] = [label for _, _, label in terms]


RSDO5 = register(
    DatasetInfo(
        name="rsdo5",
        description=(
            "RSDO5 1.1: 12 Slovenian texts in biomechanics, linguistics, chemistry and "
            "veterinary science, with about 38,000 terms marked in- or out-domain."
        ),
        languages=("sl",),
        domains=("biomechanics", "linguistics", "chemistry", "veterinary science"),
        license="CC BY-SA 4.0",
        citation=(
            "Jemec Tomazin, M., Trojar, M., Atelšek, S., Fajfar, T., Erjavec, T., & "
            "Žagar Karer, M. (2021). Corpus of term-annotated texts RSDO5 1.1. "
            "Slovenian language resource repository CLARIN.SI. "
            "http://hdl.handle.net/11356/1470"
        ),
        homepage="http://hdl.handle.net/11356/1470",
        resources=(
            Resource(
                "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1470/"
                "rsdo5.vert.zip",
                "8728570497afd46c48ae384f3c8ada9c465b005723e24cb2991da2cd8b4b08a2",
                "rsdo5.vert.zip",
            ),
        ),
        loader=load_rsdo5,
    )
)
