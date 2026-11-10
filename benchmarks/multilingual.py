"""
Run the multilingual benchmark of BENCHMARKS.md: every method of TermXtract on HTEC 2.0
(Hindi).

    python benchmarks/multilingual.py                 # everything, to benchmarks/multilingual.json
    python benchmarks/multilingual.py --methods tfidf cvalue
    python benchmarks/multilingual.py --markdown      # print the tables of a saved run

The datasets are downloaded on first use (see `termxtract.datasets`), as is a sample of
100,000 sentences of Hindi Wikipedia from the Leipzig Corpora Collection, which the
reference-corpus methods use as general-language text. The token-classification
method needs the `transformers` extra, and a GPU to finish in reasonable time.

There is no part-of-speech tagger or lemmatiser for these languages in the benchmark,
so every method reads the words as written, and candidates are sequences of up to four
words that neither start nor end with a stopword, whose words each have a letter or
digit, and that occur at least twice.

HTEC has no published split, so its documents are split here by article, with a
fixed seed: about 70% for training, 10% for validation and 20% for testing, as in
Banerjee (2025). All the methods are scored on the test documents; the supervised
methods are trained on the training documents, and token classification keeps the
epoch with the best F1 on the validation documents.
"""

import argparse
import json
import random
import re
import tarfile
import time
from collections.abc import Iterable
from pathlib import Path

from english import (
    MIN_FREQUENCY,
    REFERENCE_METHODS,
    SUPERVISED,
    TOKEN_CLASSIFICATION,
    UNSUPERVISED,
    N,
    ranking,
    scores,
    written_filter,
)
from teanga import Corpus

import termxtract
from termxtract import ATEResults, SupervisedTermExtractor, TermExtractor, datasets
from termxtract.candidates import WORD
from termxtract.datasets import base
from termxtract.evaluation import normalize_term
from termxtract.stopwords import load_stopwords

KS = (100, 500)
"""The numbers of top-ranked terms that precision, recall and F1 are reported at. The
test documents have a few hundred gold terms, so fewer than for English."""

LEIPZIG = datasets.Resource(
    "https://downloads.wortschatz-leipzig.de/corpora/hin_wikipedia_2021_100K.tar.gz",
    "bdb8cd8d52db3a974319c4c2791d3cac322aea223e71e4b204587afcff52289d",
    "hin_wikipedia_2021_100K.tar.gz",
)
"""100,000 sentences of Hindi Wikipedia (2021) from the Leipzig Corpora Collection
(Goldhahn, Eckart & Quasthoff, 2012), about 1.8 million words."""

REFERENCE_DOCUMENT = 100
"""The number of consecutive sentences of the reference corpus in each document."""

SPLIT = {"train": 0.7, "validation": 0.1, "test": 0.2}
"""The share of HTEC's documents in each part of the split."""

SEED = 0
"""The seed of the split."""

RAKE_DELIMITERS = list(".,;:!?()[]\"'।॥")
"""The punctuation that RAKE splits phrases at, with the danda and double danda."""


def load_reference(exclude: Iterable[str]) -> Corpus:
    """
    The Leipzig sample of Hindi Wikipedia as a Teanga corpus with a `tokens` layer,
    leaving out the sentences in `exclude`, such as those of the dataset.
    """
    path = base.data_home() / "leipzig" / LEIPZIG.filename
    if not path.exists():
        base._fetch(LEIPZIG, path)
    excluded = {" ".join(sentence.split()) for sentence in exclude}
    with tarfile.open(path) as archive:
        member = archive.extractfile(
            "hin_wikipedia_2021_100K/hin_wikipedia_2021_100K-sentences.txt"
        )
        assert member is not None
        lines = member.read().decode("utf-8").splitlines()
    sentences = [line.split("\t", 1)[1] for line in lines]
    sentences = [s for s in sentences if " ".join(s.split()) not in excluded]
    corpus = Corpus()
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("tokens", layer_type="span", base="text")
    for i in range(0, len(sentences), REFERENCE_DOCUMENT):
        text = "\n".join(sentences[i : i + REFERENCE_DOCUMENT])
        doc = corpus.add_doc(text=text)
        doc.tokens = [m.span() for m in re.finditer(rf"{WORD.pattern}|\S", text)]
    return corpus


def split_htec(corpus: Corpus) -> dict[str, list[str]]:
    """
    Split the documents of HTEC by article, so that the segments of an article, such
    as `99`, `99_a` and `99_b`, are in the same part. The articles are shuffled with
    `SEED`, and each goes to the part furthest below its share of `SPLIT`.
    """
    articles: dict[str, list[str]] = {}
    for doc_id in corpus.doc_ids:
        source = corpus.doc_by_id(doc_id)["source"].text[0]
        articles.setdefault(source.split("_")[0], []).append(doc_id)
    order = sorted(articles, key=int)
    random.Random(SEED).shuffle(order)
    parts: dict[str, list[str]] = {name: [] for name in SPLIT}
    total = len(corpus.doc_ids)
    for article in order:
        name = max(SPLIT, key=lambda p: SPLIT[p] * total - len(parts[p]))
        parts[name] += articles[article]
    return parts


def subset(corpus: Corpus, doc_ids: list[str]) -> Corpus:
    """A new corpus of some documents of a loaded HTEC corpus, with all their layers."""
    out = Corpus()
    for name in ("text", "source", "tokens", "sentences", "terms", "categories"):
        desc = corpus.meta[name]
        out.add_layer_meta(
            name, layer_type=desc.layer_type, base=desc.base, data=desc.data
        )
    for doc_id in doc_ids:
        doc = corpus.doc_by_id(doc_id)
        new = out.add_doc(text=doc.text, source=doc["source"].text[0])
        for name in ("tokens", "sentences", "terms"):
            new[name] = [(span[0], span[1]) for span in doc[name]]
        new["categories"] = list(doc["categories"])
    return out


def tagged_occurrences(corpus: Corpus) -> set[tuple[str, int, int]]:
    """
    The gold occurrences that token classification learns to tag, as (document,
    start, end): the annotations that cover whole words, keeping the longest where
    they overlap, and the first of two of the same length.
    """
    occurrences = set()
    for doc_id in corpus.doc_ids:
        doc = corpus.doc_by_id(doc_id)
        starts = {span[0] for span in doc.tokens}
        ends = {span[1] for span in doc.tokens}
        spans = sorted(
            {(s[0], s[1]) for s in doc.terms if s[0] in starts and s[1] in ends},
            key=lambda span: (span[0] - span[1], span),
        )
        kept: list[tuple[int, int]] = []
        for start, end in spans:
            if all(end <= a or start >= b for a, b in kept):
                kept.append((start, end))
        occurrences.update((doc_id, start, end) for start, end in kept)
    return occurrences


def mention_scores(results: ATEResults, corpus: Corpus) -> dict:
    """Precision, recall and F1 of the tagged occurrences, by exact offsets, against
    `tagged_occurrences`."""
    assert results.offsets is not None
    found = {
        (doc["doc_id"], *occurrence["offset"])
        for doc in results.offsets
        for occurrence in doc["offsets"]
    }
    gold = tagged_occurrences(corpus)
    correct = len(found & gold)
    precision = correct / len(found) if found else 0.0
    recall = correct / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if correct else 0.0
    return {
        "occurrences": len(found),
        "gold": len(gold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def unsupervised(method: str, reference: Corpus) -> TermExtractor:
    """An unsupervised extractor with the benchmark settings."""
    options: dict = {}
    if method in REFERENCE_METHODS:
        options["reference_corpus"] = reference
    if method == "rake":
        options["stoplist"] = sorted(load_stopwords("hindi"))
        options["phrase_delimiters"] = RAKE_DELIMITERS
    else:
        options.update(min_frequency=MIN_FREQUENCY, stopwords="hindi")
    return TermExtractor(method=method, n=N, words_layer="tokens", **options)


def supervised(
    method: str, train: Corpus, validation: Corpus, test: Corpus
) -> ATEResults:
    """Train a supervised extractor on the training documents and extract the terms of
    the test documents, as for the English benchmark."""
    if method == "token-classification":
        extractor = SupervisedTermExtractor(
            method=method,
            words_layer="tokens",
            stopwords="hindi",
            **TOKEN_CLASSIFICATION,
        )
        return extractor.fit(train, validation=validation).predict(test)
    gold = {normalize_term(t) for t in datasets.gold_terms(train)}
    extractor = SupervisedTermExtractor(
        method=method,
        n=N,
        words_layer="tokens",
        stopwords="hindi",
        random_state=0,
        n_jobs=-1,
    )
    return extractor.fit(train, dict.fromkeys(gold, 1)).predict(test)


def run_htec(methods: Iterable[str], out: dict, save) -> None:
    """Run the methods on HTEC, adding the scores of each to `out` and saving."""
    corpus = datasets.load("htec")
    parts = split_htec(corpus)
    train, validation, test = (
        subset(corpus, parts[name]) for name in ("train", "validation", "test")
    )
    out["split"] = {
        name: sorted(corpus.doc_by_id(d)["source"].text[0] for d in ids)
        for name, ids in parts.items()
    }
    reference = load_reference(
        s for d in corpus.doc_ids for s in corpus.doc_by_id(d).sentences.text
    )
    gold = {normalize_term(t) for t in datasets.gold_terms(test)}
    keep_written = written_filter(test)
    for method in methods:
        start = time.perf_counter()
        if method in SUPERVISED:
            results = supervised(method, train, validation, test)
            # As for English, token classification's tagged terms are not filtered
            keep = bool if method == "token-classification" else keep_written
        else:
            results = unsupervised(method, reference).extract(test)
            keep = keep_written if method == "rake" else bool
        ranked = ranking(results, keep)
        out[method] = {
            **scores(ranked, gold, KS),
            "seconds": time.perf_counter() - start,
        }
        if method == "token-classification":
            out[method]["mentions"] = mention_scores(results, test)
        save()
        print(f"htec {method}: F1@500 {out[method]['500']['f1']:.3f}", flush=True)


def markdown(results: dict) -> str:
    """The tables of BENCHMARKS.md for saved results."""
    lines = []
    for dataset, methods in results["datasets"].items():
        lines += [
            f"#### {dataset.upper()}",
            "",
            "| Method | Candidates | Gold | AvP (ATR4S) | "
            + " | ".join(f"P@{k} | F1@{k}" for k in KS)
            + " | P | R | F1 | Time (s) |",
            "|---|---:|---:|---:|" + "---:|---:|" * len(KS) + "---:|---:|---:|---:|",
        ]
        for method in UNSUPERVISED + SUPERVISED:
            if method not in methods:
                continue
            s = methods[method]
            cells = [
                f"{s[str(k)]['precision'] * 100:.1f} | {s[str(k)]['f1'] * 100:.1f}"
                for k in KS
            ]
            share = s["gold_candidates"] / s["candidates"] if s["candidates"] else 0
            every = s["all"]
            lines.append(
                f"| {method} | {s['candidates']:,} | {share:.0%} | "
                f"{s['atr4s_average_precision']:.3f} | "
                + " | ".join(cells)
                + f" | {every['precision'] * 100:.1f} | {every['recall'] * 100:.1f}"
                f" | {every['f1'] * 100:.1f} | {s['seconds']:.0f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--methods", nargs="+", default=UNSUPERVISED + SUPERVISED)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("multilingual.json")
    )
    parser.add_argument(
        "--markdown", action="store_true", help="Print the tables of a saved run."
    )
    args = parser.parse_args()

    if args.markdown:
        print(markdown(json.loads(args.output.read_text())))
        return

    results = (
        json.loads(args.output.read_text())
        if args.output.exists()
        else {"datasets": {}}
    )
    results["termxtract"] = termxtract.__version__

    def save() -> None:
        args.output.write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    run_htec(args.methods, results["datasets"].setdefault("htec", {}), save)


if __name__ == "__main__":
    main()
