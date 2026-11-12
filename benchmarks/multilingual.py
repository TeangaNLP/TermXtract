"""
Run the multilingual benchmark of BENCHMARKS.md: every method of TermXtract on HTEC 2.0
(Hindi), RSDO5 1.1 (Slovenian) and the Irish Wikipedia gold standard.

    python benchmarks/multilingual.py                 # everything, to benchmarks/multilingual.json
    python benchmarks/multilingual.py --datasets rsdo5 --methods tfidf cvalue
    python benchmarks/multilingual.py --markdown      # print the tables of a saved run

The datasets are downloaded on first use (see `termxtract.datasets`), as are samples of
Hindi and Irish Wikipedia from the Leipzig Corpora Collection, which the reference-corpus
methods use as general-language text for HTEC and the Irish gold standard. The token-classification
method needs the `transformers` extra, and a GPU to finish in reasonable time.

HTEC has no part-of-speech tags or lemmas, so on HTEC every method reads the words as
written, and candidates are sequences of up to four words that neither start nor end
with a stopword, whose words each have a letter or digit, and that occur at least twice.

HTEC has no published split, so its documents are split here by article, with a
fixed seed: about 70% for training, 10% for validation and 20% for testing, as in
Banerjee (2025). All the methods are scored on the test documents; the supervised
methods are trained on the training documents, and token classification keeps the
epoch with the best F1 on the validation documents.

RSDO5 comes with lemmas and Universal Dependencies tags, so the unsupervised methods use
the candidates of the English benchmark, with the `noun_phrase` pattern over `upos`:
sequences of up to four lemmas that match it, pass ATR4S's noise filter and occur at
least twice. Each domain is a test domain in turn, as in Tran et al. (2022): the
supervised methods train on two other domains, and token classification validates on
the fourth. The reference-corpus methods compare the test domain with the other three.
Stopwords are from the Slovenian list.

The Irish gold standard is tokenised but has no lemmas or tags, so the unsupervised
methods read the words as written, as on HTEC, over all 11 articles, with the Irish
stopwords. It is too small for a fixed split, so the supervised methods are evaluated
leaving one article out: each article is the test article in turn, token classification
validates on the next one, and the methods train on the other nine. Their results are
pooled over the 11 folds: the term sets, for precision, recall and F1, since the
rankings of 11 models cannot be merged, and for token classification the tagged
occurrences and the B, I and O tag of each token, as McCrae & Doyle (2019) report.
"""

import argparse
import json
import random
import re
import tarfile
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from english import KS as ENGLISH_KS
from english import (
    MIN_FREQUENCY,
    REFERENCE_METHODS,
    SUPERVISED,
    TOKEN_CLASSIFICATION,
    UNSUPERVISED,
    N,
    method_options,
    ranking,
    scores,
    written_filter,
)
from teanga import Corpus

import termxtract
from termxtract import ATEResults, SupervisedTermExtractor, TermExtractor, datasets
from termxtract.candidates import WORD
from termxtract.datasets import base, rsdo5
from termxtract.evaluation import normalize_term
from termxtract.stopwords import load_stopwords

KS = {"htec": (100, 500), "irish-wikipedia": (100, 500)}
"""The numbers of top-ranked terms that precision, recall and F1 are reported at, by
dataset. HTEC's test documents and the Irish gold standard have a few hundred gold
terms, so fewer than for English; the other datasets use those of the English
benchmark."""

RSDO5_FOLDS = {
    "jez": (("bim", "kem"), "vet"),
    "vet": (("bim", "kem"), "jez"),
    "kem": (("bim", "vet"), "jez"),
    "bim": (("kem", "vet"), "jez"),
}
"""For each test domain of RSDO5, the training domains and the validation domain: for
each, the first of the combinations in Table 3 of Tran et al. (2022)."""

TITLES = {
    "htec": "HTEC 2.0",
    "rsdo5-bim": "RSDO5, biomechanics",
    "rsdo5-jez": "RSDO5, linguistics",
    "rsdo5-kem": "RSDO5, chemistry",
    "rsdo5-vet": "RSDO5, veterinary science",
    "irish-wikipedia": "Irish Wikipedia",
}


LEIPZIG_HINDI = datasets.Resource(
    "https://downloads.wortschatz-leipzig.de/corpora/hin_wikipedia_2021_100K.tar.gz",
    "bdb8cd8d52db3a974319c4c2791d3cac322aea223e71e4b204587afcff52289d",
    "hin_wikipedia_2021_100K.tar.gz",
)
"""100,000 sentences of Hindi Wikipedia (2021) from the Leipzig Corpora Collection
(Goldhahn, Eckart & Quasthoff, 2012), about 1.8 million words."""

LEIPZIG_IRISH = datasets.Resource(
    "https://downloads.wortschatz-leipzig.de/corpora/gle_wikipedia_2021_30K.tar.gz",
    "605ddb1d7847d609826392bbc44177ce7b189a3bd342d8dc837e9a00c6a6853c",
    "gle_wikipedia_2021_30K.tar.gz",
)
"""30,000 sentences of Irish Wikipedia (2021) from the Leipzig Corpora Collection, about
560,000 words: its largest Irish sample."""

REFERENCE_DOCUMENT = 100
"""The number of consecutive sentences of the reference corpus in each document."""

SPLIT = {"train": 0.7, "validation": 0.1, "test": 0.2}
"""The share of HTEC's documents in each part of the split."""

SEED = 0
"""The seed of the split."""

RAKE_DELIMITERS = list(".,;:!?()[]\"'।॥")
"""The punctuation that RAKE splits phrases at, with the danda and double danda."""


def load_reference(
    resource: datasets.Resource, exclude: Callable[[str], bool]
) -> Corpus:
    """
    A Leipzig sample as a Teanga corpus with a `tokens` layer, leaving out the
    sentences for which `exclude` is true, such as those of the dataset.
    """
    path = base.data_home() / "leipzig" / resource.filename
    if not path.exists():
        base._fetch(resource, path)
    name = resource.filename.removesuffix(".tar.gz")
    with tarfile.open(path) as archive:
        member = archive.extractfile(f"{name}/{name}-sentences.txt")
        assert member is not None
        lines = member.read().decode("utf-8").splitlines()
    sentences = [line.split("\t", 1)[1] for line in lines]
    sentences = [s for s in sentences if not exclude(s)]
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
    """A new corpus of some documents of a loaded corpus, with all their layers."""
    out = Corpus()
    for name, desc in corpus.meta.items():
        out.add_layer_meta(
            name, layer_type=desc.layer_type, base=desc.base, data=desc.data
        )
    characters = [n for n, d in corpus.meta.items() if d.layer_type == "characters"]
    for doc_id in doc_ids:
        doc = corpus.doc_by_id(doc_id)
        new = out.add_doc(**{name: doc[name].text[0] for name in characters})
        for name, desc in corpus.meta.items():
            if desc.layer_type == "span":
                new[name] = [(span[0], span[1]) for span in doc[name]]
            elif desc.layer_type == "seq":
                new[name] = list(doc[name])
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


def found_occurrences(results: ATEResults) -> set[tuple[str, int, int]]:
    """The occurrences that token classification tagged, as (document, start, end)."""
    assert results.offsets is not None
    return {
        (doc["doc_id"], *occurrence["offset"])
        for doc in results.offsets
        for occurrence in doc["offsets"]
    }


def prf(found: int, gold: int, correct: int) -> dict:
    """Precision, recall and F1 from counts."""
    precision = correct / found if found else 0.0
    recall = correct / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if correct else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def mention_scores(found: set[tuple[str, int, int]], corpus: Corpus) -> dict:
    """Precision, recall and F1 of the tagged occurrences, by exact offsets, against
    `tagged_occurrences`."""
    gold = tagged_occurrences(corpus)
    return {
        "occurrences": len(found),
        "gold": len(gold),
        **prf(len(found), len(gold), len(found & gold)),
    }


def iob(corpus: Corpus, occurrences: set[tuple[str, int, int]]) -> list[str]:
    """The B, I or O tag of each token of the corpus, given the occurrences of terms."""
    tags = []
    for doc_id in corpus.doc_ids:
        doc = corpus.doc_by_id(doc_id)
        spans = [(s, e) for d, s, e in occurrences if d == doc_id]
        for start, end in doc.tokens:
            covering = [s for s, e in spans if s <= start and end <= e]
            tags.append("O" if not covering else "B" if start in covering else "I")
    return tags


def tag_scores(corpus: Corpus, found: set[tuple[str, int, int]]) -> dict:
    """Precision, recall and F1 of each of the B, I and O tags of the tokens."""
    gold = iob(corpus, tagged_occurrences(corpus))
    predicted = iob(corpus, found)
    return {
        tag: prf(
            predicted.count(tag),
            gold.count(tag),
            sum(p == g == tag for p, g in zip(predicted, gold)),
        )
        for tag in "BIO"
    }


def unsupervised(
    method: str, reference: Corpus, stopwords: str = "hindi"
) -> TermExtractor:
    """An unsupervised extractor with the benchmark settings for words as written."""
    options: dict = {}
    if method in REFERENCE_METHODS:
        options["reference_corpus"] = reference
    if method == "rake":
        options["stoplist"] = sorted(load_stopwords(stopwords))
        options["phrase_delimiters"] = RAKE_DELIMITERS
    else:
        options.update(min_frequency=MIN_FREQUENCY, stopwords=stopwords)
    return TermExtractor(**method_options(method), n=N, words_layer="tokens", **options)


def supervised(
    method: str,
    train: Corpus,
    validation: Corpus,
    test: Corpus,
    reference: Corpus,
    stopwords: str = "hindi",
) -> ATEResults:
    """Train a supervised extractor on the training documents and extract the terms of
    the test documents, as for the English benchmark. Without tags, the features
    method's candidates are every sequence of words."""
    if method == "token-classification":
        extractor = SupervisedTermExtractor(
            method=method,
            words_layer="tokens",
            stopwords=stopwords,
            **TOKEN_CLASSIFICATION,
        )
        return extractor.fit(train, validation=validation).predict(test)
    options = {"reference_corpus": reference} if method == "features" else {}
    gold = {normalize_term(t) for t in datasets.gold_terms(train)}
    extractor = SupervisedTermExtractor(
        method=method,
        n=N,
        words_layer="tokens",
        stopwords=stopwords,
        random_state=0,
        n_jobs=-1,
        **options,
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
    excluded = {
        " ".join(s.split())
        for d in corpus.doc_ids
        for s in corpus.doc_by_id(d).sentences.text
    }
    reference = load_reference(
        LEIPZIG_HINDI, lambda sentence: " ".join(sentence.split()) in excluded
    )
    gold = {normalize_term(t) for t in datasets.gold_terms(test)}
    keep_written = written_filter(test)
    for method in methods:
        start = time.perf_counter()
        if method in SUPERVISED:
            results = supervised(method, train, validation, test, reference)
            # As for English, only Roger's terms are filtered
            keep = keep_written if method == "roger" else bool
        else:
            results = unsupervised(method, reference).extract(test)
            keep = keep_written if method == "rake" else bool
        ranked = ranking(results, keep)
        out[method] = {
            **scores(ranked, gold, KS["htec"]),
            "seconds": time.perf_counter() - start,
        }
        if method == "token-classification":
            out[method]["mentions"] = mention_scores(found_occurrences(results), test)
        save()
        print(f"htec {method}: F1@500 {out[method]['500']['f1']:.3f}", flush=True)


def rsdo5_unsupervised(method: str, reference: list[Corpus]) -> TermExtractor:
    """An unsupervised extractor with the benchmark settings for RSDO5."""
    options: dict = {}
    if method in REFERENCE_METHODS:
        options["reference_corpus"] = reference
    if method == "rake":
        # RAKE reads the text itself, so it has no lemmas or tags
        options["stoplist"] = sorted(load_stopwords("slovenian"))
        options["phrase_delimiters"] = list(".,;:!?()[]\"'")
    else:
        options.update(
            lemma_layer="lemma",
            min_frequency=MIN_FREQUENCY,
            word_pattern="atr4s",
            pos_pattern="noun_phrase",
            pos_layer="upos",
            stopwords="slovenian",
        )
    return TermExtractor(**method_options(method), n=N, words_layer="tokens", **options)


def rsdo5_supervised(
    method: str, train: Corpus, validation: Corpus, test: Corpus
) -> ATEResults:
    """Train a supervised extractor on the training domains of RSDO5 and extract the
    terms of the test domain."""
    if method == "token-classification":
        extractor = SupervisedTermExtractor(
            method=method,
            words_layer="tokens",
            stopwords="slovenian",
            **TOKEN_CLASSIFICATION,
        )
        return extractor.fit(train, validation=validation).predict(test)
    # The other domains are the reference corpus of the unsupervised methods, but
    # include the training domains, so the features method has none
    options = (
        {"pos_pattern": "learned", "pos_layer": "upos"} if method == "features" else {}
    )
    gold = {normalize_term(t) for t in datasets.gold_terms(train)}
    extractor = SupervisedTermExtractor(
        method=method,
        n=N,
        words_layer="tokens",
        stopwords="slovenian",
        random_state=0,
        n_jobs=-1,
        **options,
    )
    return extractor.fit(train, dict.fromkeys(gold, 1)).predict(test)


def run_rsdo5(methods: Iterable[str], results: dict, save) -> None:
    """Run the methods on each test domain of RSDO5, adding the scores of each to
    `results["datasets"]` and saving."""
    for test_domain, (train_domains, validation_domain) in RSDO5_FOLDS.items():
        out = results["datasets"].setdefault(f"rsdo5-{test_domain}", {})
        out["fold"] = {
            "train": list(train_domains),
            "validation": validation_domain,
            "test": test_domain,
        }
        test = datasets.load("rsdo5", domains=test_domain)
        others = [d for d in rsdo5.DOMAINS if d != test_domain]
        lemma_gold = {
            normalize_term(t) for t in datasets.gold_terms(test, lemma_layer="lemma")
        }
        written_gold = {normalize_term(t) for t in datasets.gold_terms(test)}
        keep_written = written_filter(test)
        for method in methods:
            start = time.perf_counter()
            if method in SUPERVISED:
                results_ = rsdo5_supervised(
                    method,
                    datasets.load("rsdo5", domains=train_domains),
                    datasets.load("rsdo5", domains=validation_domain),
                    test,
                )
                gold = written_gold
                keep = keep_written if method == "roger" else bool
            else:
                reference = [datasets.load("rsdo5", domains=d) for d in others]
                results_ = rsdo5_unsupervised(method, reference).extract(test)
                # RAKE finds words as written, and the other methods lemmas
                gold = written_gold if method == "rake" else lemma_gold
                keep = keep_written if method == "rake" else bool
            out[method] = {
                **scores(ranking(results_, keep), gold),
                "seconds": time.perf_counter() - start,
            }
            save()
            f1 = out[method]["2000"]["f1"]
            print(f"rsdo5-{test_domain} {method}: F1@2000 {f1:.3f}", flush=True)


def run_irish(methods: Iterable[str], out: dict, save) -> None:
    """Run the methods on the Irish gold standard, adding the scores of each to `out`
    and saving."""
    corpus = datasets.load("irish-wikipedia")
    doc_ids = list(corpus.doc_ids)
    # The texts are tokenised, so compare them with the reference without spaces
    texts = ["".join(corpus.doc_by_id(d).text.split()) for d in doc_ids]
    reference = load_reference(
        LEIPZIG_IRISH,
        lambda sentence: any("".join(sentence.split()) in text for text in texts),
    )
    gold = {normalize_term(t) for t in datasets.gold_terms(corpus)}
    keep_written = written_filter(corpus)
    for method in methods:
        start = time.perf_counter()
        if method not in SUPERVISED:
            results = unsupervised(method, reference, stopwords="ga").extract(corpus)
            keep = keep_written if method == "rake" else bool
            out[method] = {
                **scores(ranking(results, keep), gold, KS["irish-wikipedia"]),
                "seconds": time.perf_counter() - start,
            }
            save()
            f1 = out[method]["500"]["f1"]
            print(f"irish-wikipedia {method}: F1@500 {f1:.3f}", flush=True)
            continue
        # Leave one article out, validating on the next, and pool the folds
        terms: set[str] = set()
        found: set[tuple[str, int, int]] = set()
        for i, test_id in enumerate(doc_ids):
            validation_id = doc_ids[(i + 1) % len(doc_ids)]
            train_ids = [d for d in doc_ids if d not in (test_id, validation_id)]
            results = supervised(
                method,
                subset(corpus, train_ids),
                subset(corpus, [validation_id]),
                subset(corpus, [test_id]),
                reference,
                stopwords="ga",
            )
            # As for English, only Roger's terms are filtered
            keep = keep_written if method == "roger" else bool
            terms |= {t["term"] for t in ranking(results, keep).terms[0]["terms"]}
            if method == "token-classification":
                found |= found_occurrences(results)
        out[method] = {
            "candidates": len(terms),
            "gold": len(gold),
            "gold_candidates": len(terms & gold),
            "all": prf(len(terms), len(gold), len(terms & gold)),
            "seconds": time.perf_counter() - start,
        }
        if method == "token-classification":
            out[method]["mentions"] = mention_scores(found, corpus)
            out[method]["tags"] = tag_scores(corpus, found)
        save()
        f1 = out[method]["all"]["f1"]
        print(f"irish-wikipedia {method}: F1 {f1:.3f}", flush=True)


def markdown(results: dict) -> str:
    """The tables of BENCHMARKS.md for saved results."""
    lines = []
    for dataset, methods in results["datasets"].items():
        ks = KS.get(dataset, ENGLISH_KS)
        lines += [
            f"#### {TITLES[dataset]}",
            "",
            "| Method | Candidates | Gold | AvP (ATR4S) | "
            + " | ".join(f"P@{k} | F1@{k}" for k in ks)
            + " | P | R | F1 | Time (s) |",
            "|---|---:|---:|---:|" + "---:|---:|" * len(ks) + "---:|---:|---:|---:|",
        ]
        for method in UNSUPERVISED + SUPERVISED:
            if method not in methods:
                continue
            s = methods[method]
            if "skipped" in s:
                lines.append(f"| {method} | {s['skipped']} |")
                continue
            # The pooled folds of the Irish gold standard have no ranking
            cells = [
                f"{s[str(k)]['precision'] * 100:.1f} | {s[str(k)]['f1'] * 100:.1f}"
                if str(k) in s
                else "– | –"
                for k in ks
            ]
            share = s["gold_candidates"] / s["candidates"] if s["candidates"] else 0
            every = s["all"]
            average = s.get("atr4s_average_precision")
            lines.append(
                f"| {method} | {s['candidates']:,} | {share:.0%} | "
                + ("–" if average is None else f"{average:.3f}")
                + " | "
                + " | ".join(cells)
                + f" | {every['precision'] * 100:.1f} | {every['recall'] * 100:.1f}"
                f" | {every['f1'] * 100:.1f} | {s['seconds']:.0f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--datasets", nargs="+", default=["htec", "rsdo5", "irish-wikipedia"]
    )
    parser.add_argument("--methods", nargs="+", default=UNSUPERVISED + SUPERVISED)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("multilingual.json")
    )
    parser.add_argument(
        "--markdown", action="store_true", help="Print the tables of a saved run."
    )
    args = parser.parse_args()

    if args.markdown:
        print(markdown(json.loads(args.output.read_text(encoding="utf-8"))))
        return

    results = (
        json.loads(args.output.read_text(encoding="utf-8"))
        if args.output.exists()
        else {"datasets": {}}
    )
    results["termxtract"] = termxtract.__version__

    def save() -> None:
        args.output.write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    if "htec" in args.datasets:
        run_htec(args.methods, results["datasets"].setdefault("htec", {}), save)
    if "rsdo5" in args.datasets:
        run_rsdo5(args.methods, results, save)
    if "irish-wikipedia" in args.datasets:
        run_irish(
            args.methods, results["datasets"].setdefault("irish-wikipedia", {}), save
        )


if __name__ == "__main__":
    main()
