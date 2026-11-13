"""
Run the few-shot benchmark of BENCHMARKS.md: in-context learning with a large language
model against fine-tuned token classification, with 5 to 30 training sentences from each
domain of ACTER 1.5, as in Banerjee, Chakravarthi & McCrae (2024).

    python benchmarks/fewshot.py                      # everything, to benchmarks/fewshot.json
    python benchmarks/fewshot.py --domains htfl --methods llm
    python benchmarks/fewshot.py --markdown           # print the table of a saved run

Each domain's sentences are shuffled with a fixed seed: the first 150 are the test
sentences and the next 10 the validation sentences, and for each of five seeds the
training sentences are a random sample of the rest. In-context learning shows each test
sentence with all its training sentences, the most similar last (`k` is the number of
training sentences), and token classification fine-tunes XLM-RoBERTa on them with the
batch size and learning rate of Banerjee (2025, Table 4.1) for 50 epochs, keeping the
epoch with the best F1 on the validation sentences. Each sentence's terms are compared with its gold
terms, and the scores are micro-averaged over the sentences.

The llm method needs a model server, as for the English benchmark, and token
classification the `transformers` extra and a GPU. A local model server may need to
unload its model before token classification has the memory of the GPU, so run the
methods one at a time with `--methods`.
"""

import argparse
import json
import random
import statistics
import time
from collections.abc import Iterable
from pathlib import Path

from english import LLM_MODEL, TOKEN_CLASSIFICATION, llm
from teanga import Corpus

import termxtract
from termxtract import SupervisedTermExtractor, datasets, evaluate
from termxtract.datasets.acter import DOMAINS

SIZES = (5, 15, 25, 30)
"""The numbers of training sentences."""

SEEDS = range(5)
"""The seeds of the samples of training sentences."""

TEST, VALIDATION = 150, 10
"""The numbers of test and validation sentences of each domain."""

FINE_TUNING = {
    # domain: (batch size, learning rate), from Banerjee (2025, Table 4.1), with the
    # batch size multiplied by the steps of gradient accumulation
    "corp": (8, 1.14e-4),
    "equi": (16, 1.53e-4),
    "htfl": (24, 1.92e-5),
    "wind": (24, 1.01e-4),
}

METHODS = ["llm", "token-classification"]

TITLES = {
    "corp": "Corruption",
    "equi": "Dressage",
    "htfl": "Heart failure",
    "wind": "Wind energy",
}


def sentences(domain: str) -> list[tuple[str, list, list]]:
    """The distinct sentences of an ACTER domain, each with its tokens and gold terms
    as offsets within it, in a fixed random order."""
    corpus = datasets.load("acter", domains=domain, include_named_entities=True)
    found: dict[str, tuple[str, list, list]] = {}
    for doc_id in corpus.doc_ids:
        doc = corpus.doc_by_id(doc_id)
        tokens = [(s, e) for s, e in doc["tokens"]]
        terms = [(s, e) for s, e in doc["terms"].indexes("text")]
        for start, end in doc["sentences"].indexes("text"):
            # Teanga names a document by its text, so a repeated sentence is kept once
            found.setdefault(
                doc.text[start:end],
                (
                    doc.text[start:end],
                    [(s - start, e - start) for s, e in tokens if start <= s < end],
                    [(s - start, e - start) for s, e in terms if start <= s < end],
                ),
            )
    shuffled = list(found.values())
    random.Random(0).shuffle(shuffled)
    return shuffled


def as_corpus(items: Iterable[tuple[str, list, list]]) -> Corpus:
    """A Teanga corpus with one document per sentence."""
    corpus = Corpus()
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("tokens", layer_type="span", base="text")
    corpus.add_layer_meta("sentences", layer_type="span", base="text")
    corpus.add_layer_meta("terms", layer_type="span", base="text")
    for text, tokens, terms in items:
        doc = corpus.add_doc(text)
        doc.tokens = tokens
        doc.sentences = [(0, len(text))]
        doc.terms = sorted(terms)
    return corpus


def extract(
    method: str, domain: str, seed: int, train: Corpus, validation: Corpus, test: Corpus
):
    """Train a method on the training sentences and extract the terms of the test
    sentences."""
    if method == "llm":
        return llm(k=len(train.doc_ids)).fit(train).predict(test)
    batch_size, learning_rate = FINE_TUNING[domain]
    # Train for 50 epochs without stopping early: with few training sentences and a
    # low learning rate, the F1 on the validation sentences can stay at 0 for many
    # epochs before the model tags anything
    options = {**TOKEN_CLASSIFICATION, "epochs": 50, "patience": None}
    options["random_state"] = seed
    extractor = SupervisedTermExtractor(
        method=method,
        words_layer="tokens",
        batch_size=batch_size,
        learning_rate=learning_rate,
        **options,
    )
    return extractor.fit(train, validation=validation).predict(test)


def run(domain: str, methods: Iterable[str], out: dict, save) -> None:
    """Run the methods on a domain, adding their scores to `out` and saving."""
    items = sentences(domain)
    test = as_corpus(items[:TEST])
    validation = as_corpus(items[TEST : TEST + VALIDATION])
    pool = items[TEST + VALIDATION :]
    for method in methods:
        scores = out.setdefault(method, {})
        for size in SIZES:
            runs = []
            start = time.perf_counter()
            for seed in SEEDS:
                train = as_corpus(random.Random(seed).sample(pool, size))
                results = extract(method, domain, seed, train, validation, test)
                e = evaluate(results, test, average="micro")
                runs.append({"precision": e.precision, "recall": e.recall, "f1": e.f1})
            scores[str(size)] = {
                "runs": runs,
                "seconds": time.perf_counter() - start,
            }
            save()
            f1 = statistics.mean(r["f1"] for r in runs)
            print(f"{domain} {method} {size}: F1 {f1:.3f}", flush=True)


def markdown(results: dict) -> str:
    """The table of BENCHMARKS.md for saved results."""
    domains = [d for d in DOMAINS if d in results["domains"]]
    lines = [
        "| Training sentences | Method | "
        + " | ".join(f"{TITLES[d]} P / R / F1" for d in domains)
        + " |",
        "|---:|---|" + "---:|" * len(domains),
    ]

    def cell(runs: list[dict]) -> str:
        return (
            " / ".join(
                f"{statistics.mean(r[m] for r in runs) * 100:.1f}"
                for m in ("precision", "recall", "f1")
            )
            + f" (±{statistics.stdev(r['f1'] for r in runs) * 100:.1f})"
        )

    for size in SIZES:
        for method in METHODS:
            row = [
                results["domains"][d].get(method, {}).get(str(size)) for d in domains
            ]
            if any(row):
                lines.append(
                    f"| {size} | {method} | "
                    + " | ".join(cell(r["runs"]) if r else "" for r in row)
                    + " |"
                )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--domains", nargs="+", default=list(DOMAINS))
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("fewshot.json")
    )
    parser.add_argument(
        "--markdown", action="store_true", help="Print the table of a saved run."
    )
    args = parser.parse_args()

    if args.markdown:
        print(markdown(json.loads(args.output.read_text())))
        return

    results = (
        json.loads(args.output.read_text()) if args.output.exists() else {"domains": {}}
    )
    results["termxtract"] = termxtract.__version__
    results["llm_model"] = LLM_MODEL

    def save() -> None:
        args.output.write_text(json.dumps(results, indent=2) + "\n")

    for domain in args.domains:
        run(domain, args.methods, results["domains"].setdefault(domain, {}), save)


if __name__ == "__main__":
    main()
