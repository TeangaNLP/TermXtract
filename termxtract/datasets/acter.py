"""
ACTER, the Annotated Corpora for Term Extraction Research, version 1.5.
"""

import zipfile
from collections.abc import Iterable
from pathlib import Path

from teanga import Corpus

from .base import CATEGORY_LAYER, GOLD_LAYER, DatasetInfo, Resource, register

_COMMIT = "f05b09e985cad37eeaa8daa8b3f383197aa5324e"
"""The commit of the v1.5 tag of https://github.com/AylaRT/ACTER."""

LANGUAGES = ("en", "fr", "nl")
"""The languages of ACTER: English, French and Dutch."""

DOMAINS = ("corp", "equi", "htfl", "wind")
"""The domains of ACTER: corruption, dressage (equitation), heart failure and wind
energy."""

TERM_LABELS = ("Specific_Term", "Common_Term", "OOD_Term")
"""The labels of terms, as opposed to named entities."""

NAMED_ENTITY = "Named_Entity"
"""The label of named entities."""

SPLITS = {"train": ("corp", "equi", "wind"), "test": ("htfl",)}
"""The domains of each split, as in the TermEval 2020 shared task."""

SOURCE_LAYER = "source"
"""The characters layer holding the name of the file each document comes from, such as
`htfl_en_001`."""

SENTENCES_LAYER = "sentences"
"""The span layer of the sentences of each document."""

IOB_LAYER = "iob"
"""The sequence layer, based on `tokens`, of ACTER's IOB labels."""


def load_acter(
    path: str | Path,
    *,
    split: str | None = None,
    language: str = "en",
    domains: str | Iterable[str] | None = None,
    include_named_entities: bool = False,
    labels: Iterable[str] | None = None,
) -> Corpus:
    """
    Load ACTER from the archive of its GitHub repository.

    Each annotated text is a document, with these layers:

    - `text`, and `source`, the name of its file (such as `htfl_en_001`).
    - `tokens` and `sentences`, from ACTER's tokenised texts.
    - `terms` (`GOLD_LAYER`): every occurrence in the document of a term in ACTER's
      list of tokenised annotations for the language and domain, including nested
      terms, within a sentence and at token boundaries, ignoring case. ACTER lists its
      terms per corpus, not per occurrence, so as in TermEval a term is gold wherever
      it occurs.
    - `categories` (`CATEGORY_LAYER`): the label of each term, one of `TERM_LABELS` or
      `NAMED_ENTITY`. A term labelled differently in different contexts has its most
      frequent label.
    - `iob`: ACTER's sequential IOB labels, one per token. These give the longest
      annotation where annotations are nested, and include partly annotated tokens and
      interrupted terms. They follow `include_named_entities` but not `labels`.

    Args:
        path (Union[str, Path]): The ACTER archive.
        split (Optional[str]): `"train"` for the corruption, dressage and wind energy
            domains, `"test"` for heart failure (as in TermEval 2020), or None for all
            four.
        language (str): `"en"`, `"fr"` or `"nl"`.
        domains (Optional[Union[str, Iterable[str]]]): Domains to load, from `DOMAINS`.
            With `split`, only the domains in both are loaded.
        include_named_entities (bool): Count named entities as gold terms.
        labels (Optional[Iterable[str]]): The labels of the gold terms, from
            `TERM_LABELS` and, with `include_named_entities`, `NAMED_ENTITY`. None for
            all of them.

    Returns:
        Corpus: The documents, in order of domain and file name.

    Raises:
        ValueError: If an option is unknown or no domain is left to load.
    """
    if language not in LANGUAGES:
        raise ValueError(
            f"Unknown ACTER language: {language!r}. Use one of "
            f"{', '.join(map(repr, LANGUAGES))}."
        )
    chosen = _choose_domains(split, domains)
    allowed = TERM_LABELS + ((NAMED_ENTITY,) if include_named_entities else ())
    keep = set(
        allowed if labels is None else [labels] if isinstance(labels, str) else labels
    )
    if keep - set(allowed):
        raise ValueError(
            f"Unknown ACTER labels: {', '.join(sorted(keep - set(allowed)))}. Use "
            f"{', '.join(allowed)}"
            + (
                ""
                if include_named_entities
                else ", or pass include_named_entities=True"
            )
            + "."
        )

    corpus = Corpus()
    corpus.add_layer_meta("text")
    corpus.add_layer_meta(SOURCE_LAYER)
    corpus.add_layer_meta("tokens", layer_type="span", base="text")
    corpus.add_layer_meta(SENTENCES_LAYER, layer_type="span", base="text")
    corpus.add_layer_meta(GOLD_LAYER, layer_type="span", base="text")
    corpus.add_layer_meta(
        CATEGORY_LAYER, layer_type="seq", base=GOLD_LAYER, data=list(allowed)
    )
    corpus.add_layer_meta(
        IOB_LAYER, layer_type="seq", base="tokens", data=["B", "I", "O"]
    )

    nes = "_nes" if include_named_entities else ""
    iob_dir = "with" if include_named_entities else "without"
    with zipfile.ZipFile(path) as archive:
        for domain in chosen:
            root = f"ACTER-{_COMMIT}/{language}/{domain}/annotated/"
            annotations = root + "annotations/"
            terms = {
                tuple(term.split()): label
                for term, label in _read_tsv(
                    archive,
                    f"{annotations}unique_annotation_lists/"
                    f"{domain}_{language}_tokenised_terms{nes}.tsv",
                )
                if label in keep
            }
            longest = max(map(len, terms), default=0)
            texts = sorted(
                name
                for name in archive.namelist()
                if name.startswith(root + "texts/") and name.endswith(".txt")
            )
            for name in texts:
                stem = name.rsplit("/", 1)[1][: -len(".txt")]
                text = archive.read(name).decode("utf-8")
                sentences = [
                    line.split()
                    for line in archive.read(f"{root}texts_tokenised/{stem}.txt")
                    .decode("utf-8")
                    .splitlines()
                    if line.strip()
                ]
                tokens = _align(text, [token for s in sentences for token in s], stem)
                iob = [
                    tag
                    for _, tag in _read_tsv(
                        archive,
                        f"{annotations}sequential_annotations/iob_annotations/"
                        f"{iob_dir}_named_entities/{stem}_seq_terms{nes}.tsv",
                    )
                ]
                if len(iob) != len(tokens):
                    raise ValueError(
                        f"ACTER's IOB labels for {stem} do not match its tokens."
                    )

                sentence_spans = []
                found = []
                start = 0
                for sentence in sentences:
                    end = start + len(sentence)
                    sentence_spans.append((tokens[start][0], tokens[end - 1][1]))
                    lowered = [token.lower() for token in sentence]
                    for i in range(len(sentence)):
                        for n in range(1, min(longest, len(sentence) - i) + 1):
                            label = terms.get(tuple(lowered[i : i + n]))
                            if label is not None:
                                found.append(
                                    (
                                        tokens[start + i][0],
                                        tokens[start + i + n - 1][1],
                                        label,
                                    )
                                )
                    start = end
                found.sort()

                doc = corpus.add_doc(text=text, source=stem)
                doc.tokens = tokens
                doc[SENTENCES_LAYER] = sentence_spans
                doc[GOLD_LAYER] = [(s, e) for s, e, _ in found]
                doc[CATEGORY_LAYER] = [label for _, _, label in found]
                doc[IOB_LAYER] = iob
    return corpus


def _choose_domains(
    split: str | None, domains: str | Iterable[str] | None
) -> list[str]:
    """The domains to load, in the order of `DOMAINS`."""
    chosen = set(DOMAINS if split is None else SPLITS[split])
    if domains is not None:
        wanted = {domains} if isinstance(domains, str) else set(domains)
        unknown = wanted - set(DOMAINS)
        if unknown:
            raise ValueError(
                f"Unknown ACTER domains: {', '.join(sorted(unknown))}. Use "
                f"{', '.join(DOMAINS)}."
            )
        chosen &= wanted
        if not chosen:
            raise ValueError(
                "No ACTER domains to load: "
                + (
                    "domains is empty."
                    if split is None
                    else f"the {split!r} split has only {', '.join(SPLITS[split])}."
                )
            )
    return [domain for domain in DOMAINS if domain in chosen]


def _read_tsv(archive: zipfile.ZipFile, name: str) -> list[tuple[str, str]]:
    """The non-empty lines of a two-column TSV file in the archive."""
    rows = []
    for line in archive.read(name).decode("utf-8").splitlines():
        if line.strip():
            first, second = line.split("\t")
            rows.append((first, second.strip()))
    return rows


def _align(text: str, tokens: list[str], stem: str) -> list[tuple[int, int]]:
    """
    Find the character offsets of ACTER's tokens in the original text.

    The tokeniser drops double quotes, even inside tokens (`"BA"-onderdelen` becomes
    `BA-onderdelen`), so quotes in the text are skipped.
    """
    spans = []
    pos = 0
    for token in tokens:
        while pos < len(text) and (text[pos].isspace() or text[pos] == '"'):
            pos += 1
        start = pos
        for char in token:
            while pos < len(text) and text[pos] == '"' and char != '"':
                pos += 1
            if pos >= len(text) or text[pos] != char:
                raise ValueError(
                    f"ACTER's token {token!r} does not match the text of {stem} at "
                    f"offset {start}."
                )
            pos += 1
        spans.append((start, pos))
    return spans


ACTER = register(
    DatasetInfo(
        name="acter",
        description=(
            "ACTER 1.5: texts in English, French and Dutch from four domains, with "
            "manually annotated terms and named entities."
        ),
        languages=LANGUAGES,
        domains=("corruption", "dressage", "heart failure", "wind energy"),
        license="CC BY-NC-SA 4.0",
        citation=(
            "Rigouts Terryn, A., Hoste, V., & Lefever, E. (2020). In no uncertain "
            "terms: a dataset for monolingual and multilingual automatic term "
            "extraction from comparable corpora. Language Resources and Evaluation, "
            "54(2), 385-418. https://doi.org/10.1007/s10579-019-09453-9"
        ),
        homepage="https://github.com/AylaRT/ACTER",
        resources=(
            Resource(
                f"https://github.com/AylaRT/ACTER/archive/{_COMMIT}.zip",
                "a5852de72e1c5a1f346ab80673f50173b8f5478d28109ce3b906d816d10ac238",
                "ACTER-1.5.zip",
            ),
        ),
        loader=load_acter,
        splits=tuple(SPLITS),
    )
)
