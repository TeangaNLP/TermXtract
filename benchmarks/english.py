"""
Run the English benchmark of BENCHMARKS.md: every method of TermXtract on ACTER 1.5,
the ACL RD-TEC 2.0 and the GENIA term corpus.

    python benchmarks/english.py                      # everything, to benchmarks/english.json
    python benchmarks/english.py --datasets acter --methods tfidf cvalue
    python benchmarks/english.py --markdown           # print the tables of a saved run

The datasets are downloaded on first use (see `termxtract.datasets`), as is the Brown
corpus, which the reference-corpus methods use as general-language text. Corpora without
part-of-speech tags and lemmas (ACTER, GENIA and Brown) are tagged with spaCy's
`en_core_web_sm`, which must be installed:

    pip install spacy
    python -m spacy download en_core_web_sm

The tags are cached next to the datasets. The token-classification method needs the
`transformers` extra, and a GPU to finish in reasonable time.

The unsupervised methods use the candidates of ATR4S (Astrakhantsev, 2018), whose
results BENCHMARKS.md compares with: sequences of up to four lemmas, counting only the
occurrences that neither start nor end with a stopword, match `POS_PATTERN`, and pass
ATR4S's noise filter (`word_pattern="atr4s"`), and that occur at least twice. The
supervised methods read the words as written, and keep the candidates that
occur at least twice and consist of words with a letter or digit.
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

N = 4
"""The maximum number of words in a candidate."""

MIN_FREQUENCY = 2
"""The minimum number of occurrences of a candidate in the corpus."""

KS = (100, 500, 1000, 2000)
"""The numbers of top-ranked terms that precision, recall and F1 are reported at."""

POS_PATTERN = "(<NNS?>|<JJ>|<NNPS?>)*<NNS?|NNPS?>"
"""
ATR4S's part-of-speech pattern, `(NN(S)?|JJ|NNP|NN(S?)IN)*(NN(S)?)`, over Penn Treebank
tags, adapted to spaCy's tags. The prepositions are left out, since the tags of ATR4S's
noise filter rule them out, and candidates may also end in a proper noun: spaCy tags
many biomedical names, such as HIV and AP-1, as proper nouns, and without them a third
of the occurrences of GENIA's gold terms would not be counted.
"""

TAGGER = "en_core_web_sm"
"""The spaCy model that tags the corpora without part-of-speech tags and lemmas."""

DATASETS: dict[str, tuple[dict[str, Any], dict[str, Any] | None]] = {
    # name: (options of the test corpus, options of the training corpus)
    "acter": (
        {"split": "test", "include_named_entities": True},
        {"split": "train", "include_named_entities": True},
    ),
    "acl-rd-tec": ({}, None),
    "genia": ({}, None),
}

VALIDATION: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    # name: (options of the training corpus, of the validation corpus) for token
    # classification, added to the options of the training corpus. As in Lang et al.
    # (2021), it trains on corruption and wind energy and validates on dressage.
    "acter": ({"domains": ("corp", "wind")}, {"domains": "equi"}),
}

TOKEN_CLASSIFICATION = {"epochs": 10, "patience": 2, "random_state": 0}
"""The options of token classification: train for up to 10 epochs, keep the best on
the validation corpus, and stop after 2 epochs without improvement."""

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
SUPERVISED = ["roger", "token-classification"]

SKIP: dict[tuple[str, str], str] = {}
"""Methods that are not run on a dataset, as (dataset, method), with the reason."""

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


def tag(corpus: Corpus, name: str) -> Corpus:
    """
    Add `pos` (Penn Treebank) and `lemma` layers, based on `tokens`, to a corpus without
    them, with spaCy's `TAGGER`. The tags are cached under the data directory.
    """
    if "pos" in corpus.meta and "lemma" in corpus.meta:
        return corpus
    import spacy
    from spacy.tokens import Doc

    nlp = spacy.load(TAGGER, exclude=["parser", "ner"])
    model = f"{TAGGER}-{nlp.meta['version']}"
    path = base.data_home() / "benchmark-tags" / f"{name}-{model}.json"
    if path.exists():
        tags = json.loads(path.read_text())
    else:
        tags = {}
        for doc_id in corpus.doc_ids:
            words = list(corpus.doc_by_id(doc_id)["tokens"].text)
            tagged = list(nlp(Doc(nlp.vocab, words=words))) if words else []
            tags[doc_id] = [[t.tag_ for t in tagged], [t.lemma_ for t in tagged]]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(tags))
    corpus.add_layer_meta("pos", layer_type="seq", base="tokens", data="string")
    corpus.add_layer_meta("lemma", layer_type="seq", base="tokens", data="string")
    for doc_id in corpus.doc_ids:
        doc = corpus.doc_by_id(doc_id)
        doc.pos, doc.lemma = tags[doc_id]
    return corpus


def ngram_counts(corpus: Corpus, layer: str = "tokens") -> Counter[str]:
    """The number of occurrences of each lowercased n-gram of up to `N` words, read
    from the `tokens` layer or from a sequence layer based on it, such as `lemma`."""
    counts: Counter[str] = Counter()
    for doc_id in corpus.doc_ids:
        doc = corpus.doc_by_id(doc_id)
        tokens = doc.tokens.text if layer == "tokens" else doc[layer]
        words = [normalize_term(str(w)) for w in tokens]
        for n in range(1, N + 1):
            counts.update(" ".join(words[i : i + n]) for i in range(len(words) - n + 1))
    return counts


def written_filter(corpus: Corpus) -> Callable[[str], bool]:
    """The candidates for the supervised methods: n-grams of words as written that occur
    at least `MIN_FREQUENCY` times and whose words have a letter or digit."""
    counts = ngram_counts(corpus)

    def keep(term: str) -> bool:
        return counts[term] >= MIN_FREQUENCY and all(
            any(c.isalnum() for c in word) for word in term.split()
        )

    return keep


def ranking(results: ATEResults, keep: Callable[[str], bool]) -> ATEResults:
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
    Average precision as ATR4S reports it: over the first K terms, where K is the number
    of gold terms among the candidates, averaging the precision at each gold term found.
    A perfect ranking scores 1, and a random one about the share of candidates that are
    gold terms.
    """
    k = len(set(ranked) & gold)
    hits = 0
    total = 0.0
    for rank, term in enumerate(ranked[:k], start=1):
        if term in gold:
            hits += 1
            total += hits / rank
    return total / hits if hits else 0.0


def scores(results: ATEResults, gold: set[str], ks: Iterable[int] = KS) -> dict:
    """The scores of a single-document corpus ranking against the gold terms, at each
    of `ks` and over all of it."""
    ranked = [term["term"] for term in results.terms[0]["terms"]]
    out: dict = {
        "candidates": len(ranked),
        "gold": len(gold),
        "gold_candidates": len(set(ranked) & gold),
        "atr4s_average_precision": atr4s_average_precision(ranked, gold),
        "average_precision": evaluate(results, gold).average_precision,
    }
    for k in (*ks, None):
        e = evaluate(results, gold, top_k=k)
        out["all" if k is None else str(k)] = {
            "precision": e.precision,
            "recall": e.recall,
            "f1": e.f1,
        }
    return out


def unsupervised(method: str, reference: Corpus | None):
    """An unsupervised extractor with the benchmark settings."""
    options: dict = {}
    if method in REFERENCE_METHODS:
        options["reference_corpus"] = reference
    if method == "rake":
        # RAKE reads the text itself, so it has no lemmas
        options["stoplist"] = sorted(load_stopwords("english"))
        options["phrase_delimiters"] = list(".,;:!?()[]\"'")
    else:
        options.update(
            lemma_layer="lemma",
            min_frequency=MIN_FREQUENCY,
            word_pattern="atr4s",
            pos_pattern=POS_PATTERN,
            pos_layer="pos",
        )
    return TermExtractor(method=method, n=N, words_layer="tokens", **options)


def supervised(dataset: str, method: str, train: Corpus, test: Corpus) -> ATEResults:
    """Train a supervised extractor on the gold terms of the training corpus and
    extract the terms of the test corpus. Roger takes every other candidate as a
    non-term, and token classification learns from each annotated occurrence."""
    if method == "token-classification":
        train_options = DATASETS[dataset][1]
        assert train_options is not None
        train_extra, validation_extra = VALIDATION[dataset]
        extractor = SupervisedTermExtractor(
            method=method, words_layer="tokens", **TOKEN_CLASSIFICATION
        )
        return extractor.fit(
            datasets.load(dataset, **train_options, **train_extra),
            validation=datasets.load(dataset, **train_options, **validation_extra),
        ).predict(test)
    gold = {normalize_term(t) for t in datasets.gold_terms(train)}
    # Roger's evolutionary search is random, so fix its seed; its runs are independent
    extractor = SupervisedTermExtractor(
        method=method, n=N, words_layer="tokens", random_state=0, n_jobs=-1
    )
    return extractor.fit(train, dict.fromkeys(gold, 1)).predict(test)


def run(
    dataset: str,
    methods: Iterable[str],
    reference: Corpus,
    out: dict,
    save: Callable[[], None],
) -> None:
    """Run the methods on a dataset, adding the scores of each to `out` and saving."""
    test_options, train_options = DATASETS[dataset]
    test = tag(datasets.load(dataset, **test_options), dataset)
    train = datasets.load(dataset, **train_options) if train_options else None
    # The unsupervised methods find lemmas, and the supervised ones words as written
    lemma_gold = {
        normalize_term(t) for t in datasets.gold_terms(test, lemma_layer="lemma")
    }
    written_gold = {normalize_term(t) for t in datasets.gold_terms(test)}
    keep_written = written_filter(test)
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
            results = supervised(dataset, method, train, test)
            gold = written_gold
            # Token classification tags terms rather than ranking candidates, so its
            # terms are not filtered, as in Lang et al. (2021)
            ranked = ranking(
                results, bool if method == "token-classification" else keep_written
            )
        elif method == "rake":
            results = unsupervised(method, reference).extract(test)
            gold = written_gold
            ranked = ranking(results, keep_written)
        else:
            results = unsupervised(method, reference).extract(test)
            gold = lemma_gold
            ranked = ranking(results, bool)
        out[method] = {**scores(ranked, gold), "seconds": time.perf_counter() - start}
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
            "| Method | Candidates | Gold | AvP (ATR4S) | "
            + " | ".join(f"P@{k} | F1@{k}" for k in KS)
            + " | Time (s) |",
            "|---|---:|---:|---:|" + "---:|---:|" * len(KS) + "---:|",
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
            share = s["gold_candidates"] / s["candidates"] if s["candidates"] else 0
            lines.append(
                f"| {method} | {s['candidates']:,} | {share:.0%} | "
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

    reference = tag(load_brown(), "brown")
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
