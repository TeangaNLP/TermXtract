"""
Run the English benchmark of BENCHMARKS.md: every method of TermXtract on ACTER 1.5,
the ACL RD-TEC 2.0 and the GENIA term corpus.

    python benchmarks/english.py                      # everything, to benchmarks/english.json
    python benchmarks/english.py --datasets acter --methods tfidf cvalue
    python benchmarks/english.py --markdown           # print the tables of a saved run

The datasets are downloaded on first use (see `termxtract.datasets`), as is the Brown
corpus, which the reference-corpus methods use as general-language text. The
token-classification method needs the `transformers` extra, and a GPU to finish in
reasonable time.

The settings follow ATR4S (Astrakhantsev, 2018), whose results BENCHMARKS.md compares
with: candidates of up to four words, which neither start nor end with a stopword,
consist of words with at least one letter or digit, and occur at least twice in the
corpus, and, where the dataset has part-of-speech tags, match the `noun_phrase` pattern.
"""

import argparse
import json
import re
import time
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from teanga import Corpus

import termxtract
from termxtract import ATEResults, SupervisedTermExtractor, TermExtractor, datasets
from termxtract.datasets import base
from termxtract.evaluation import evaluate, normalize_term
from termxtract.stopwords import load_stopwords
from termxtract.supervised.words import tokenize_teanga

N = 4
"""The maximum number of words in a candidate."""

MIN_FREQUENCY = 2
"""The minimum number of occurrences of a candidate in the corpus."""

KS = (100, 500, 1000, 2000)
"""The numbers of top-ranked terms that precision, recall and F1 are reported at."""

DATASETS: dict[str, tuple[dict[str, Any], dict[str, Any] | None, str | None]] = {
    # name: (options of the test corpus, options of the training corpus, POS pattern)
    "acter": (
        {"split": "test", "include_named_entities": True},
        {"split": "train", "include_named_entities": True},
        None,
    ),
    "acl-rd-tec": ({}, None, "noun_phrase"),
    "genia": ({}, None, None),
}

TITLES = {"acter": "ACTER", "acl-rd-tec": "ACL RD-TEC 2.0", "genia": "GENIA"}

UNSUPERVISED = [
    "tfidf",
    "ridf",
    "cvalue",
    "basic",
    "combobasic",
    "domaincoherence",
    "relevance",
    "domainpertinence",
    "weirdness",
    "topicmodeling",
    "nmf",
    "rake",
]
REFERENCE_METHODS = {"relevance", "domainpertinence", "weirdness"}
SUPERVISED = ["adaboost", "roger", "token-classification"]

SKIP = {
    ("genia", "ridf"): "not run: RIDF takes hours on GENIA (see the known problems)",
    (
        "genia",
        "nmf",
    ): "not run: NMF runs out of memory on GENIA (see the known problems)",
}

BROWN = datasets.Resource(
    "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/"
    "brown.zip",
    "9b275f9b3b95d7bd66ccfb7cd259f445a13bbe5d1f4107aba09fd3e8364bafa6",
    "brown.zip",
)
"""The Brown corpus (Francis & Kučera, 1979), from the NLTK data repository: about one
million words of American English, distributed with permission to redistribute."""


def load_brown() -> Corpus:
    """The Brown corpus as a Teanga corpus with a `tokens` layer, one document per file."""
    path = base.data_home() / "brown" / BROWN.filename
    if not path.exists():
        base._fetch(BROWN, path)
    corpus = Corpus()
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("tokens", layer_type="span", base="text")
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if not re.fullmatch(r"brown/c[a-r]\d\d", name):
                continue
            # Tokens are written word/TAG, separated by whitespace
            words = [
                token.rsplit("/", 1)[0]
                for token in archive.read(name).decode("utf-8").split()
            ]
            spans = []
            offset = 0
            for word in words:
                spans.append((offset, offset + len(word)))
                offset += len(word) + 1
            doc = corpus.add_doc(text=" ".join(words))
            doc.tokens = spans
    return corpus


def ngram_counts(corpus: Corpus) -> Counter[str]:
    """The number of occurrences of each lowercased n-gram of up to `N` tokens."""
    counts: Counter[str] = Counter()
    for doc_id in corpus.doc_ids:
        words = [normalize_term(w) for w in corpus.doc_by_id(doc_id).tokens.text]
        for n in range(1, N + 1):
            counts.update(" ".join(words[i : i + n]) for i in range(len(words) - n + 1))
    return counts


def candidate_filter(corpus: Corpus):
    """The ATR4S-style filter of candidates for a corpus, on top of the stopwords."""
    counts = ngram_counts(corpus)

    def keep(term: str) -> bool:
        return counts[term] >= MIN_FREQUENCY and all(
            any(c.isalnum() for c in word) for word in term.split()
        )

    return keep


def ranking(results: ATEResults, keep) -> ATEResults:
    """The corpus ranking of some results, as a single document, filtered by `keep`."""
    best: dict[str, float] = {}
    for doc in results.terms:
        for term in doc["terms"]:
            key = normalize_term(term["term"])
            if keep(key) and term["score"] > best.get(key, float("-inf")):
                best[key] = term["score"]
    ranked = sorted(best.items(), key=lambda item: -item[1])
    return ATEResults(
        corpus=results.corpus,
        terms=[
            {
                "doc_id": "corpus",
                "terms": [{"term": t, "score": s} for t, s in ranked],
            }
        ],
    )


def atr4s_average_precision(ranked: list[str], gold: set[str]) -> float:
    """
    Average precision as ATR4S reports it: of the first K terms, where K is the number
    of gold terms among the candidates, so a perfect ranking scores 1.
    """
    k = len(set(ranked) & gold)
    hits = 0
    total = 0.0
    for rank, term in enumerate(ranked[:k], start=1):
        if term in gold:
            hits += 1
            total += hits / rank
    return total / k if k else 0.0


def scores(results: ATEResults, gold: set[str]) -> dict:
    """The scores of a single-document corpus ranking against the gold terms."""
    ranked = [term["term"] for term in results.terms[0]["terms"]]
    out: dict = {
        "candidates": len(ranked),
        "gold": len(gold),
        "gold_candidates": len(set(ranked) & gold),
        "atr4s_average_precision": atr4s_average_precision(ranked, gold),
        "average_precision": evaluate(results, gold).average_precision,
    }
    for k in (*KS, None):
        e = evaluate(results, gold, top_k=k)
        out["all" if k is None else str(k)] = {
            "precision": e.precision,
            "recall": e.recall,
            "f1": e.f1,
        }
    return out


def unsupervised(method: str, pos_pattern: str | None, reference: Corpus | None):
    """An unsupervised extractor with the benchmark settings."""
    options: dict = {}
    if method in REFERENCE_METHODS:
        options["reference_corpus"] = reference
    if method == "rake":
        options["stoplist"] = sorted(load_stopwords("english"))
        options["phrase_delimiters"] = list(".,;:!?()[]\"'")
    return TermExtractor(
        method=method, n=N, words_layer="tokens", pos_pattern=pos_pattern, **options
    )


def supervised(method: str, train: Corpus, test: Corpus) -> ATEResults:
    """
    Train a supervised extractor on one corpus and score the candidates of another.

    The wrapper's `extract` trains and predicts on the same corpus, so this uses the
    extractors directly.
    """
    gold = {normalize_term(t) for t in datasets.gold_terms(train)}
    # The extractors have different methods, so the type is that of each branch
    extractor: Any = SupervisedTermExtractor(
        method=method, n=N, words_layer="tokens"
    ).extractor
    if method == "token-classification":
        extractor.train_terms_teanga(train, dict.fromkeys(gold, 1))
        return extractor.extract_terms_teanga(test)
    if method == "roger":
        # Roger only learns from labelled candidates, so label every candidate
        labels = {term: int(term in gold) for term in ngram_counts(train)}
    else:
        labels = dict.fromkeys(gold, 1)
    extractor.train(tokenize_teanga(train, "tokens"), labels)
    predicted = extractor.predict(tokenize_teanga(test, "tokens"))
    return ATEResults(
        corpus=test,
        terms=[
            {
                "doc_id": "corpus",
                "terms": [{"term": t, "score": float(s)} for t, s in predicted.items()],
            }
        ],
    )


def run(
    dataset: str,
    methods: Iterable[str],
    reference: Corpus,
    out: dict,
    save: Callable[[], None],
) -> None:
    """Run the methods on a dataset, adding the scores of each to `out` and saving."""
    test_options, train_options, pos_pattern = DATASETS[dataset]
    test = datasets.load(dataset, **test_options)
    train = datasets.load(dataset, **train_options) if train_options else None
    gold = {normalize_term(t) for t in datasets.gold_terms(test)}
    keep = candidate_filter(test)
    for method in methods:
        if method in SUPERVISED and train is None:
            continue
        if (dataset, method) in SKIP:
            out[method] = {"skipped": SKIP[dataset, method]}
            save()
            print(f"{dataset} {method}: {SKIP[dataset, method]}", flush=True)
            continue
        start = time.perf_counter()
        if method in SUPERVISED:
            assert train is not None
            results = supervised(method, train, test)
        else:
            results = unsupervised(method, pos_pattern, reference).extract(test)
        out[method] = {
            # Token classification tags terms rather than ranking candidates, so its
            # terms are not filtered, as in Lang et al. (2021)
            **scores(
                ranking(results, keep if method != "token-classification" else bool),
                gold,
            ),
            "seconds": time.perf_counter() - start,
        }
        save()
        f1 = out[method]["2000"]["f1"]
        print(f"{dataset} {method}: F1@2000 {f1:.3f}", flush=True)


def markdown(results: dict) -> str:
    """The tables of BENCHMARKS.md for saved results."""
    lines = []
    for dataset, methods in results["datasets"].items():
        lines += [
            f"#### {TITLES[dataset]}",
            "",
            "| Method | Candidates | AvP (ATR4S) | "
            + " | ".join(f"P@{k} | F1@{k}" for k in KS)
            + " | Time (s) |",
            "|---|---:|---:|" + "---:|---:|" * len(KS) + "---:|",
        ]
        for method in UNSUPERVISED + SUPERVISED:
            if method not in methods:
                continue
            s = methods[method]
            if "skipped" in s:
                lines.append(f"| {method} | {s['skipped']} |")
                continue
            cells = [
                f"{s[str(k)]['precision'] * 100:.1f} | {s[str(k)]['f1'] * 100:.1f}"
                for k in KS
            ]
            lines.append(
                f"| {method} | {s['candidates']:,} | "
                f"{s['atr4s_average_precision']:.3f} | "
                + " | ".join(cells)
                + f" | {s['seconds']:.0f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", default=UNSUPERVISED + SUPERVISED)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("english.json")
    )
    parser.add_argument(
        "--markdown", action="store_true", help="Print the tables of a saved run."
    )
    args = parser.parse_args()

    if args.markdown:
        print(markdown(json.loads(args.output.read_text())))
        return

    reference = load_brown()
    # Add to a saved run, so methods and datasets can be run separately
    results = (
        json.loads(args.output.read_text())
        if args.output.exists()
        else {"datasets": {}}
    )
    results["termxtract"] = termxtract.__version__

    def save() -> None:
        args.output.write_text(json.dumps(results, indent=2) + "\n")

    for dataset in args.datasets:
        run(
            dataset,
            args.methods,
            reference,
            results["datasets"].setdefault(dataset, {}),
            save,
        )


if __name__ == "__main__":
    main()
