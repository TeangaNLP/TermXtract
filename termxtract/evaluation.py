from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TypeAlias

from teanga import Corpus

from .utils import ATEResults

Gold: TypeAlias = (
    Corpus | Iterable[str] | Iterable[Iterable[str]] | Mapping[str, Iterable[str]]
)
"""
A gold standard:

- a collection of terms for the whole corpus, such as a set or a list of strings,
- one collection of terms per document, in the order of the documents,
- a dictionary from document IDs to collections of terms, or
- a Teanga corpus with a span layer of gold terms.
"""

AVERAGES = ("corpus", "micro", "macro")
"""The ways `evaluate` can compare terms; see `evaluate`."""


def normalize_term(term: str) -> str:
    """
    Normalize a term for matching: lowercase it and collapse runs of whitespace.

    Args:
        term (str): A term.

    Returns:
        str: The normalized term.
    """
    return " ".join(term.lower().split())


@dataclass
class Evaluation:
    """
    Scores of extracted terms against a gold standard, from `evaluate`.

    Attributes:
        precision (float): The fraction of extracted terms that are gold terms.
        recall (float): The fraction of gold terms that were extracted.
        f1 (float): The harmonic mean of precision and recall.
        true_positives (int): The number of extracted terms that are gold terms.
        false_positives (int): The number of extracted terms that are not gold terms.
        false_negatives (int): The number of gold terms that were not extracted.
        average_precision (float): The mean of the precision at the rank of each gold
            term in the corpus ranking, counting gold terms that were not extracted as
            zero.
        r_precision (float): The precision of the first R terms of the corpus ranking,
            where R is the number of gold terms.
        precision_at_k (Dict[int, float]): The precision of the first K terms of the
            corpus ranking, for each K. If fewer than K terms were extracted, the
            missing terms count as wrong.
    """

    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    average_precision: float
    r_precision: float
    precision_at_k: dict[int, float] = field(default_factory=dict)


def evaluate(
    results: ATEResults,
    gold: Gold,
    average: str = "corpus",
    top_k: int | None = None,
    ks: Iterable[int] = (10, 100, 1000),
    term_lengths: Collection[int] | None = None,
    gold_layer: str = "terms",
    normalize: Callable[[str], str] = normalize_term,
) -> Evaluation:
    """
    Score extracted terms against a gold standard.

    Terms match if they are equal after `normalize`, which by default lowercases them and
    collapses whitespace. There is no credit for partial matches: if the gold standard
    has "heart failure patients", extracting "heart failure" is a false positive. Nested
    gold terms each count as a separate term.

    Precision, recall and F1 compare sets of terms, in one of three ways:

    - `"corpus"`: the distinct terms extracted from the whole corpus against the
      distinct gold terms, as in the TermEval shared task.
    - `"micro"`: the terms of each document against that document's gold terms, adding
      up the true positives, false positives and false negatives of all documents.
    - `"macro"`: the same per-document comparison, averaging the precision, recall and
      F1 of the documents.

    The ranking scores (average precision, R-precision and precision at K) always use the
    corpus ranking, where each term is ranked by its highest score in any document, and
    the gold terms of the whole corpus. With per-document gold terms, only the documents
    in the results count.

    Args:
        results (ATEResults): The extracted terms.
        gold (Gold): The gold standard. `"micro"` and `"macro"` need per-document gold
            terms. Documents are matched to the results by ID for a dictionary or a Teanga
            corpus, and by position for a list of collections.
        average (str): `"corpus"`, `"micro"` or `"macro"`.
        top_k (Optional[int]): If given, only the first `top_k` terms of the corpus
            ranking (for `"corpus"`) or of each document (otherwise) count as
            extracted.
        ks (Iterable[int]): The values of K for `precision_at_k`.
        term_lengths (Optional[Collection[int]]): If given, only extracted and gold terms
            with these numbers of words are scored, for example `{1}` for single-word
            terms.
        gold_layer (str): For a Teanga gold standard, the span layer of gold terms.
        normalize (Callable[[str], str]): The function applied to every term before
            matching.

    Returns:
        Evaluation: The scores.

    Raises:
        ValueError: If `average` is unknown, `"micro"` or `"macro"` is used with gold
            terms for the whole corpus, or the per-document gold terms do not match
            the documents of the results.
    """
    if average not in AVERAGES:
        raise ValueError(
            f"Unknown average: {average!r}. Use one of {', '.join(map(repr, AVERAGES))}."
        )

    def keep(term: str) -> bool:
        return term_lengths is None or len(term.split()) in term_lengths

    doc_ids = [doc["doc_id"] for doc in results.terms]
    corpus_gold, doc_gold = _read_gold(gold, doc_ids, gold_layer, normalize)
    corpus_gold = {term for term in corpus_gold if keep(term)}
    ranking = [term for term in _corpus_ranking(results, normalize) if keep(term)]

    if average == "corpus":
        extracted = set(ranking[:top_k])
        tp = len(extracted & corpus_gold)
        fp, fn = len(extracted) - tp, len(corpus_gold) - tp
        precision, recall, f1 = _prf(tp, fp, fn)
    else:
        if doc_gold is None:
            raise ValueError(
                f"average={average!r} needs gold terms for each document, but the gold "
                "standard is a single collection of terms for the whole corpus."
            )
        tp = fp = fn = 0
        doc_scores = []
        for doc, gold_terms in zip(results.terms, doc_gold):
            doc_ranking = _dedupe(normalize(t["term"]) for t in doc["terms"])
            extracted = set([term for term in doc_ranking if keep(term)][:top_k])
            gold_terms = {term for term in gold_terms if keep(term)}
            doc_tp = len(extracted & gold_terms)
            doc_fp, doc_fn = len(extracted) - doc_tp, len(gold_terms) - doc_tp
            tp, fp, fn = tp + doc_tp, fp + doc_fp, fn + doc_fn
            doc_scores.append(_prf(doc_tp, doc_fp, doc_fn))
        if average == "micro":
            precision, recall, f1 = _prf(tp, fp, fn)
        else:
            n = len(doc_scores)
            precision, recall, f1 = (
                (sum(scores[i] for scores in doc_scores) / n if n else 0.0)
                for i in range(3)
            )

    return Evaluation(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        average_precision=average_precision(ranking, corpus_gold),
        r_precision=precision_at(ranking, corpus_gold, len(corpus_gold)),
        precision_at_k={k: precision_at(ranking, corpus_gold, k) for k in ks},
    )


def precision_at(ranking: list[str], gold: Collection[str], k: int) -> float:
    """
    Get the precision of the first `k` terms of a ranking.

    Args:
        ranking (List[str]): Terms, from best to worst.
        gold (Collection[str]): The gold terms.
        k (int): The number of terms to score. If the ranking is shorter, the missing
            terms count as wrong.

    Returns:
        float: The fraction of the first `k` terms that are gold terms, or 0 if `k` is 0.
    """
    if k <= 0:
        return 0.0
    return sum(term in gold for term in ranking[:k]) / k


def average_precision(ranking: list[str], gold: Collection[str]) -> float:
    """
    Get the average precision of a ranking.

    This is the sum over the gold terms in the ranking of the precision at their rank,
    divided by the number of gold terms, so gold terms missing from the ranking count as
    zero.

    Args:
        ranking (List[str]): Terms, from best to worst.
        gold (Collection[str]): The gold terms.

    Returns:
        float: The average precision, or 0 if there are no gold terms.
    """
    if not gold:
        return 0.0
    hits = 0
    total = 0.0
    for rank, term in enumerate(ranking, start=1):
        if term in gold:
            hits += 1
            total += hits / rank
    return total / len(gold)


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall and F1 from counts, taking 0/0 as 0."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _dedupe(terms: Iterable[str]) -> list[str]:
    """The terms in order, without repeats."""
    return list(dict.fromkeys(terms))


def _corpus_ranking(results: ATEResults, normalize: Callable[[str], str]) -> list[str]:
    """Rank each distinct term by its best score in any document; ties keep first-seen order."""
    best: dict[str, float] = {}
    for doc in results.terms:
        for t in doc["terms"]:
            term = normalize(t["term"])
            best[term] = max(best.get(term, t["score"]), t["score"])
    return sorted(best, key=lambda term: best[term], reverse=True)


def _read_gold(
    gold: Gold,
    doc_ids: list[str],
    gold_layer: str,
    normalize: Callable[[str], str],
) -> tuple[set[str], list[set[str]] | None]:
    """
    Normalize a gold standard.

    Returns the gold terms of the whole corpus and, if the gold standard has them, the
    gold terms of each document of the results, in order.
    """
    by_doc: Mapping[str, Iterable[str]] | None = None
    if isinstance(gold, Corpus):
        if gold_layer not in gold.meta:
            raise ValueError(f"The gold Teanga corpus has no {gold_layer!r} layer.")
        by_doc = {
            doc_id: gold.doc_by_id(doc_id)[gold_layer].text for doc_id in gold.doc_ids
        }
    elif isinstance(gold, Mapping):
        by_doc = gold
    elif isinstance(gold, str):
        raise TypeError(
            "The gold standard must be a collection of terms, not a string."
        )

    if by_doc is not None:
        missing = [doc_id for doc_id in doc_ids if doc_id not in by_doc]
        if missing:
            raise ValueError(
                f"The gold standard has no terms for the documents {missing[:5]!r}"
                + (" ..." if len(missing) > 5 else "")
                + ". Documents are matched by ID."
            )
        doc_gold = [{normalize(term) for term in by_doc[doc_id]} for doc_id in doc_ids]
        return set().union(*doc_gold), doc_gold

    items = list(gold)
    if all(isinstance(item, str) for item in items):
        return {normalize(term) for term in items}, None
    if len(items) != len(doc_ids):
        raise ValueError(
            f"The gold standard has terms for {len(items)} documents, but the results "
            f"have {len(doc_ids)}."
        )
    doc_gold = [{normalize(term) for term in terms} for terms in items]
    return set().union(*doc_gold), doc_gold
