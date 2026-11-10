"""
Convert HTEC 2.0, the Hindi Term Extraction Corpus, from the JSON files of its
annotation tool to the Teanga YAML file that `termxtract.datasets.load("htec")`
downloads.

    python scripts/htec_to_teanga.py output_json.zip htec-2.0.yaml

The input is the archive of `output_json/*.json`, one file per document, each
`{"document_id", "text", "annotations": [{"term", "category", "start_offset",
"end_offset"}]}`. The output has one document per file, in order of document ID, with
these layers:

- `text`, and `source`, the document ID (such as `99_a`), without the stray space of
  `8 `. Documents with the same number are segments of one Wikipedia article.
- `terms`: every annotation, as published, sorted by offset.
- `categories`: the category of each term, one of `CATEGORIES`.

The categories are normalised to the names of Table 6.1 of Banerjee (2025): the
annotations spell the institutions category three ways. `output_file.json` is left
out: it is not a text but a list of terms, one per line.
"""

import argparse
import json
import re
import zipfile

from teanga import Corpus

CATEGORIES = (
    "Ambiguous",
    "Educational Institutions, Governing bodies, Think Tanks and Research Institutes",
    "Degrees, Disciplines and different stages of education",
    "Educationists, Learners and Researchers",
    "Education Related Policy and Regulatory Frameworks",
    "Mode of Dissemination",
    "Education Technology and Equipment",
)
"""The categories of Table 6.1, in its order."""

SPELLINGS = {
    "Educational Institutions , Governing bodies, Think Tanks and Research Institutes": (
        CATEGORIES[1]
    ),
    "Educational Institutions and frameworks, Governing bodies, Think Tanks and "
    "Research Institutes": CATEGORIES[1],
}
"""Other spellings of the categories in the annotations."""

TERM_LIST = "output_file.json"


def document_order(name: str) -> tuple[int, int, str]:
    """Sort document IDs by number, then by segment: `99`, `99_a`, …, `99_z`, `99_aa`."""
    match = re.fullmatch(r"(\d+)(?:_([a-z]+))?", name)
    if match is None:
        raise ValueError(f"Unexpected HTEC document ID: {name!r}")
    segment = match.group(2) or ""
    return int(match.group(1)), len(segment), segment


def convert(archive_path: str) -> Corpus:
    """Read the annotation archive as a Teanga corpus."""
    corpus = Corpus()
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("source")
    corpus.add_layer_meta("terms", layer_type="span", base="text")
    corpus.add_layer_meta(
        "categories", layer_type="seq", base="terms", data=list(CATEGORIES)
    )
    with zipfile.ZipFile(archive_path) as archive:
        records = [
            json.loads(archive.read(name).decode("utf-8"))
            for name in archive.namelist()
            if name.endswith(".json") and not name.endswith("/" + TERM_LIST)
        ]
    for record in records:
        record["document_id"] = record["document_id"].strip()
    for record in sorted(records, key=lambda r: document_order(r["document_id"])):
        text = record["text"]
        terms = []
        for annotation in record["annotations"]:
            start, end = annotation["start_offset"], annotation["end_offset"]
            if text[start:end] != annotation["term"]:
                raise ValueError(
                    f"{record['document_id']}: {annotation['term']!r} is not at "
                    f"{start}-{end}."
                )
            category = SPELLINGS.get(annotation["category"], annotation["category"])
            if category not in CATEGORIES:
                raise ValueError(f"Unknown HTEC category: {category!r}")
            terms.append((start, end, category))
        terms.sort(key=lambda t: (t[0], -t[1], t[2]))
        doc = corpus.add_doc(text=text, source=record["document_id"])
        doc.terms = [(start, end) for start, end, _ in terms]
        doc.categories = [category for _, _, category in terms]
    return corpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("archive", help="The archive of output_json/*.json")
    parser.add_argument("output", help="The Teanga YAML file to write")
    args = parser.parse_args()
    convert(args.archive).to_yaml(args.output)


if __name__ == "__main__":
    main()
