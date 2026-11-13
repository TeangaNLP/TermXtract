"""
Run the re-ranking benchmark of BENCHMARKS.md: AdaText and SemRe-Rank applied to
unsupervised methods on the ACL RD-TEC 2.0 and the GENIA term corpus, as in Zhang,
Petrak & Maynard (2018) and Zhang, Gao & Ciravegna (2018).

    python benchmarks/rerank.py                       # everything, to benchmarks/rerank.json
    python benchmarks/rerank.py --datasets acl-rd-tec --methods cvalue
    python benchmarks/rerank.py --markdown            # print the tables of a saved run

The base methods are run as in the English benchmark (`benchmarks/english.py`, whose
setup and requirements apply), and each ranking is re-ranked with seeds of two kinds:

- "verified": the gold terms among the `SEEDS` most frequent candidates of the method,
  standing in for SemRe-Rank's seeds, which a person picked from the most frequent
  candidates.
- "unsupervised": all of the `SEEDS` most frequent candidates, as in the unsupervised
  variant of SemRe-Rank.

AdaText is run with each threshold of `THRESHOLDS`, and SemRe-Rank with each minimum
similarity of `REL_MINS`. The rankings are scored by their
precision at each of `KS`, and by the mean of these, as in the papers. The words are
embedded by `ENCODER`, which takes a few minutes on a CPU.
"""

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

from english import ENCODER, TITLES, load_brown, ranking, tag, unsupervised
from teanga import Corpus

import termxtract
from termxtract import ATEResults, datasets, evaluate
from termxtract.embeddings import Encoder
from termxtract.evaluation import normalize_term
from termxtract.rerank import AdaText, Reranker, SemReRank

DATASETS = ["acl-rd-tec", "genia"]

METHODS = ["tfidf", "cvalue", "basic", "combobasic", "weirdness", "relevance"]
"""The base methods: those of TermXtract that the papers re-ranked, except RAKE, whose
scores in the English benchmark's setup are all 0."""

KS = (50, 100, 500, 1000, 2000)
"""The numbers of top-ranked terms that precision is reported at, as in the papers."""

SEEDS = 200
"""The number of most frequent candidates that the seeds are taken from: the larger
of the two that SemRe-Rank tried."""

THRESHOLDS = (0.5, 0.6, 0.7, 0.8)
"""AdaText's similarity thresholds. The paper tried 0.5 to 0.85."""

REL_MINS = (0.3, 0.4, 0.5)
"""SemRe-Rank's minimum similarities. The paper used 0.5 with word2vec, under which most
pairs of words are more similar than with the sentence encoder."""


def precisions(results: ATEResults, gold: set[str]) -> dict[str, float]:
    """The precision of a single-document corpus ranking at each of `KS`, and their
    mean."""
    out = {str(k): evaluate(results, gold, top_k=k).precision for k in KS}
    out["mean"] = statistics.mean(out.values())
    return out


def rerankers(
    encoder: Encoder, verified: list[str], frequent: list[str]
) -> dict[str, Reranker]:
    """The re-rankers of the benchmark, by name."""
    out: dict[str, Reranker] = {}
    for kind, seeds in (("verified", verified), ("unsupervised", frequent)):
        for rel_min in REL_MINS:
            out[f"semrerank-{rel_min}-{kind}"] = SemReRank(
                seeds=seeds, rel_min=rel_min, encoder=encoder
            )
        for threshold in THRESHOLDS:
            out[f"adatext-{threshold}-{kind}"] = AdaText(
                seeds=seeds, threshold=threshold, encoder=encoder
            )
    return out


def run(
    dataset: str,
    methods: list[str],
    reference: Corpus,
    encoder: Encoder,
    out: dict,
    save: Callable[[], None],
) -> None:
    """Re-rank the methods on a dataset, adding the scores to `out` and saving."""
    test = tag(datasets.load(dataset), dataset)
    lemma_gold = {
        normalize_term(t) for t in datasets.gold_terms(test, lemma_layer="lemma")
    }
    for method in methods:
        extractor = unsupervised(method, reference)
        results = extractor.extract(test)
        gold = lemma_gold
        docs = [
            [text for _, _, text in extractor.extractor.teanga_words(test.doc_by_id(d))]
            for d in test.doc_ids
        ]
        lower = [[word.lower() for word in doc] for doc in docs]
        terms = list(
            dict.fromkeys(
                term["term"].lower() for doc in results.terms for term in doc["terms"]
            )
        )
        frequent = Reranker.most_frequent(lower, terms, SEEDS)
        verified = [term for term in frequent if normalize_term(term) in gold]
        scores = out.setdefault(method, {})
        scores["base"] = precisions(ranking(results, bool), gold)
        scores["seeds"] = {"verified": len(verified), "unsupervised": len(frequent)}
        print(f"{dataset} {method}: {scores['base']['mean']:.3f}", flush=True)
        for name, reranker in rerankers(encoder, verified, frequent).items():
            start = time.perf_counter()
            reranked = ranking(reranker.rerank(results, docs), bool)
            scores[name] = {
                **precisions(reranked, gold),
                "seconds": time.perf_counter() - start,
            }
            print(f"{dataset} {method} {name}: {scores[name]['mean']:.3f}", flush=True)
        save()


def markdown(results: dict) -> str:
    """The tables of BENCHMARKS.md for saved results: the mean precision of each base
    method and re-ranker, with the gain over the base method."""
    lines = []
    for dataset, methods in results["datasets"].items():
        names = [
            name
            for name in next(iter(methods.values()))
            if name not in ("base", "seeds")
        ]
        lines += [
            f"#### {TITLES[dataset]}",
            "",
            "| Re-ranker | " + " | ".join(METHODS) + " |",
            "|---|" + "---:|" * len(METHODS),
            "| base | "
            + " | ".join(f"{methods[m]['base']['mean'] * 100:.1f}" for m in METHODS)
            + " |",
            "| verified seeds | "
            + " | ".join(str(methods[m]["seeds"]["verified"]) for m in METHODS)
            + " |",
        ]
        for name in names:
            cells = []
            for m in METHODS:
                gain = (methods[m][name]["mean"] - methods[m]["base"]["mean"]) * 100
                cells.append(f"{methods[m][name]['mean'] * 100:.1f} ({gain:+.1f})")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("rerank.json")
    )
    parser.add_argument(
        "--markdown", action="store_true", help="Print the tables of a saved run."
    )
    args = parser.parse_args()

    if args.markdown:
        print(markdown(json.loads(args.output.read_text())))
        return

    reference = tag(load_brown(), "brown")
    encoder = Encoder(ENCODER)
    results = (
        json.loads(args.output.read_text())
        if args.output.exists()
        else {"datasets": {}}
    )
    results["termxtract"] = termxtract.__version__
    results["encoder"] = ENCODER

    def save() -> None:
        args.output.write_text(json.dumps(results, indent=2) + "\n")

    for dataset in args.datasets:
        run(
            dataset,
            args.methods,
            reference,
            encoder,
            results["datasets"].setdefault(dataset, {}),
            save,
        )


if __name__ == "__main__":
    main()
