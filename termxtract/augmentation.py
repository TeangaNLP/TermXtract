"""
Synthetic training data for supervised term extraction, as in Banerjee (2025, ch. 5):
a large language model or a lexicon from Wikipedia writes new sentences from a sample of
the gold sentences, and the terms of the new sentences are their annotations.
"""

import math
import re
from collections.abc import Iterable

import numpy as np
from teanga import Corpus

from .datasets.base import GOLD_LAYER
from .evaluation import normalize_term
from .layers import WORDS_LAYERS, find_layer
from .llm import LLMClient
from .supervised.llm import (
    BULLET,
    PIECE,
    Sentence,
    find_term,
    gold_spans,
    labelled_spans,
    split_teanga,
    string_sentences,
)

METHODS = ("term", "context", "combined", "wiki")
"""The augmentation methods: LLM-TermDA, LLM-ContextDA, LLM-CombinedDA and Wiki-TermDA."""

SAMPLINGS = ("diversity", "random")
"""The ways of choosing the gold sentences to augment."""

PROMPTS = {
    "combined": (
        "Given these sentences where concepts related to {domain} are tagged by <tag>, "
        "generate {k} sentences while minimizing data distribution shift. Make sure "
        "that the sentences are diverse and different from those provided below. "
        "Please make sure that you don't use any terms from the provided sentences. "
        "Annotate the concepts belonging to the domain of {domain} with <tag> in the "
        "generated sentences:\n{sentences}"
    ),
    "term": (
        "Given below are sentences in which concepts related to {domain} are tagged by "
        "<tag>, replace the annotated with different concepts that are related to "
        "{domain}. While replacing the concepts be careful to not disturb the "
        "syntactic and semantic coherence of the input. Please generate {k} diverse "
        "sentences and make sure all generated sentences are in lower case:\n"
        "{sentences}"
    ),
    "context": (
        "Given a list of terms that occur in the sentences below:\n{terms}\n"
        "Sentences are:\n{sentences}\n"
        "Given these terms and the sentences, generate {k} sentences that use these "
        "terms. The terms are tagged with <tag>. In the generated sentences tagging is "
        "not required. All sentences in lower case, make sure that each sentence on a "
        "new line and the list is not numbered."
    ),
}
"""The prompts of the LLM methods (Banerjee, 2025, Table 5.1). `{domain}` is the
domain, `{k}` the number of sentences to write, `{sentences}` the sampled gold
sentences, one per line with their terms in `<tag>` tags, and `{terms}` their terms."""

TAG = re.compile(r"<tag>(.*?)</tag>", re.IGNORECASE | re.DOTALL)
"""A term tagged by the model."""

QUOTES = "\"'`“”‘’"
"""Quotes stripped from both ends of a sentence written by the model."""

Example = tuple[str, list[tuple[int, int]]]
"""A sentence and the offsets of its terms within it."""


def augment(
    corpus: Corpus | list[str],
    labels: dict[str, int] | None = None,
    method: str = "term",
    n: int = 50,
    client: LLMClient | None = None,
    domain: str | None = None,
    lexicon: dict[str, list[str]] | None = None,
    per_prompt: int = 10,
    sampling: str = "diversity",
    seeds: int = 40,
    random_state: int | None = None,
    max_words: int = 10,
    words_layer: str | None = None,
    terms_layer: str = GOLD_LAYER,
    sentences_layer: str | None = None,
) -> Corpus | tuple[list[str], dict[str, int]]:
    """
    Add synthetic sentences to a training corpus, as in Banerjee (2025, ch. 5).

    A sample of the gold sentences, the seeds, is chosen, and new sentences are written
    from them by one of four methods:

    - "term" (LLM-TermDA): the model replaces the terms of the seeds with other terms
      of the domain.
    - "context" (LLM-ContextDA): the model writes new sentences that use the terms of
      the seeds.
    - "combined" (LLM-CombinedDA): the model writes new sentences with new terms.
    - "wiki" (Wiki-TermDA): each term of a seed is replaced by a similar term from a
      lexicon, such as one from `termxtract.wikipedia.lexicon`.

    The model marks the terms of the sentences it writes with `<tag>` tags, except for
    "context", whose sentences are annotated with the terms of the seeds they contain.
    Sentences without terms, with unbalanced tags or that repeat a sentence are dropped.

    Args:
        corpus (Union[Corpus, List[str]]): The training corpus.
        labels (Optional[Dict[str, int]]): Dictionary with terms as keys and binary
            labels {0, 1}: needed for a list of strings, whose sentences are annotated
            with the terms labelled 1 they contain. None for a Teanga corpus takes the
            gold terms of `terms_layer`.
        method (str): "term", "context", "combined" or "wiki".
        n (int): The number of synthetic sentences to write. Fewer are written if the
            model or the lexicon runs out of new ones.
        client (Optional[LLMClient]): The model of the LLM methods. It must be able to
            answer at length: `max_tokens` of `OpenAIClient` should allow about 100
            tokens for each sentence of `per_prompt`.
        domain (Optional[str]): The domain, as written in the prompts of the LLM
            methods, such as "heart failure".
        lexicon (Optional[Dict[str, List[str]]]): For "wiki", the replacements of each
            term, in lower case.
        per_prompt (int): The number of sentences each prompt asks for. The seeds are
            shuffled for each prompt, so that the prompts differ.
        sampling (str): "diversity" for the seeds that together have the most distinct
            terms, as in Banerjee (2025), or "random".
        seeds (int): The number of seeds; all the gold sentences with terms if there are
            fewer.
        random_state (Optional[int]): Seed of the sampling, the order of the seeds in
            each prompt and the replacements of "wiki".
        max_words (int): For a list of strings, the maximum number of words of a
            labelled term.
        words_layer (Optional[str]): For Teanga input, the span layer holding the words,
            or None for `words` or, failing that, `tokens`.
        terms_layer (str): For Teanga input without labels, the span layer of the gold
            terms.
        sentences_layer (Optional[str]): For Teanga input, the span layer of the
            sentences, or None for `sentences` or, if there is none, splitting after
            each word that is ".", "!" or "?".

    Returns:
        Union[Corpus, Tuple[List[str], Dict[str, int]]]: For a Teanga corpus, a new
        corpus with a document for each gold sentence and each synthetic sentence, with
        the words layer, a `sentences` layer and the terms layer. For a list of strings,
        the strings followed by the synthetic sentences, and the labels with each term
        of the synthetic sentences labelled 1.

    Raises:
        ValueError: If the method or the sampling is unknown, a list of strings has no
            labels, or the method lacks its client, domain or lexicon.
    """
    if method not in METHODS:
        raise ValueError(
            f"Unknown method: {method!r}. Use one of {', '.join(METHODS)}."
        )
    if sampling not in SAMPLINGS:
        raise ValueError(
            f"Unknown sampling: {sampling!r}. Use one of {', '.join(SAMPLINGS)}."
        )
    if method == "wiki":
        if lexicon is None:
            raise ValueError("The 'wiki' method needs a lexicon.")
    elif client is None or domain is None:
        raise ValueError(f"The {method!r} method needs a client and a domain.")
    if per_prompt < 1:
        raise ValueError("per_prompt must be at least 1.")
    teanga = isinstance(corpus, Corpus)
    if labels is None and not teanga:
        raise ValueError(
            "A list of strings has no gold terms, so augment needs labels."
        )

    if teanga:
        assert isinstance(corpus, Corpus)
        layer = find_layer(corpus.meta, words_layer, WORDS_LAYERS, "words_layer")
        documents = split_teanga(corpus, layer, sentences_layer)
        sentences = [s for doc in documents for s in doc]
        if labels is None:
            spans = [
                s for doc in gold_spans(corpus, documents, terms_layer) for s in doc
            ]
        else:
            terms = {normalize_term(t) for t, label in labels.items() if label == 1}
            spans = [labelled_spans(s, terms, max_words) for s in sentences]
    else:
        assert isinstance(corpus, list) and labels is not None
        terms = {normalize_term(t) for t, label in labels.items() if label == 1}
        sentences = [s for doc in corpus for s in string_sentences(doc)]
        spans = [labelled_spans(s, terms, max_words) for s in sentences]
    gold = [
        (sentence.text, outermost(s for s in found if s[1] <= len(sentence.text)))
        for sentence, found in zip(sentences, spans)
    ]

    rng = np.random.default_rng(random_state)
    chosen = sample([g for g in gold if g[1]], seeds, sampling, rng)
    seen = {text.lower() for text, _ in gold}
    if method == "wiki":
        assert lexicon is not None
        synthetic = replace_terms(chosen, lexicon, n, rng, seen)
    else:
        assert client is not None and domain is not None
        synthetic = generate(client, method, chosen, domain, n, per_prompt, rng, seen)

    if not teanga:
        assert isinstance(corpus, list) and labels is not None
        new_labels = dict(labels)
        for text, found in synthetic:
            for start, stop in found:
                new_labels.setdefault(text[start:stop], 1)
        return corpus + [text for text, _ in synthetic], new_labels

    out = Corpus()
    out.add_layer_meta("text")
    out.add_layer_meta(layer, layer_type="span", base="text")
    out.add_layer_meta("sentences", layer_type="span", base="text")
    out.add_layer_meta(terms_layer, layer_type="span", base="text")
    added = set()
    for sentence, found in zip(sentences, spans):
        if sentence.text not in added:
            added.add(sentence.text)
            inside = [s for s in found if s[1] <= len(sentence.text)]
            _add(out, sentence.text, sentence.words, inside, layer, terms_layer)
    for text, found in synthetic:
        if text not in added:
            added.add(text)
            words = [m.span() for m in PIECE.finditer(text)]
            _add(out, text, words, found, layer, terms_layer)
    return out


def _add(
    corpus: Corpus,
    text: str,
    words: list[tuple[int, int]],
    terms: list[tuple[int, int]],
    words_layer: str,
    terms_layer: str,
) -> None:
    """Add a sentence to a corpus as a document."""
    doc = corpus.add_doc(text)
    doc[words_layer] = words
    doc["sentences"] = [(0, len(text))]
    doc[terms_layer] = sorted(set(terms))


def outermost(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Drop the spans that overlap a longer one, or an earlier one as long, so that the
    terms can be tagged.

    Args:
        spans (Iterable[Tuple[int, int]]): The spans.

    Returns:
        List[Tuple[int, int]]: The spans kept, in order.
    """
    kept: list[tuple[int, int]] = []
    for start, stop in sorted(set(spans), key=lambda s: (s[0] - s[1], s[0])):
        if all(stop <= a or start >= b for a, b in kept):
            kept.append((start, stop))
    return sorted(kept)


def sample(
    gold: list[Example], size: int, sampling: str, rng: np.random.Generator
) -> list[Example]:
    """
    Choose the gold sentences to augment.

    Args:
        gold (List[Example]): The gold sentences with terms.
        size (int): The number to choose.
        sampling (str): "diversity" to choose, one at a time, the sentence with the most
            terms not yet chosen and, of those, the most terms, as in Banerjee (2025,
            §5.1.1); or "random".
        rng (np.random.Generator): The random number generator of "random".

    Returns:
        List[Example]: The chosen sentences, in the order of `gold` for "diversity".
    """
    if len(gold) <= size:
        return list(gold)
    if sampling == "random":
        return [gold[i] for i in sorted(rng.choice(len(gold), size, replace=False))]
    terms = [{text[a:b].lower() for a, b in spans} for text, spans in gold]
    covered: set[str] = set()
    chosen: list[int] = []
    left = set(range(len(gold)))
    while len(chosen) < size:
        best = min(left, key=lambda i: (-len(terms[i] - covered), -len(terms[i]), i))
        chosen.append(best)
        left.remove(best)
        covered |= terms[best]
    return [gold[i] for i in sorted(chosen)]


def tag(text: str, spans: list[tuple[int, int]]) -> str:
    """
    Mark the terms of a sentence with `<tag>` tags.

    Args:
        text (str): The sentence.
        spans (List[Tuple[int, int]]): The offsets of its terms, which do not overlap.

    Returns:
        str: The sentence with each term in tags.
    """
    parts = []
    end = 0
    for start, stop in sorted(spans):
        parts += [text[end:start], "<tag>", text[start:stop], "</tag>"]
        end = stop
    parts.append(text[end:])
    return "".join(parts)


def untag(line: str) -> Example | None:
    """
    Read a sentence written by the model with its terms in `<tag>` tags.

    Args:
        line (str): The sentence.

    Returns:
        Optional[Example]: The sentence without the tags and the offsets of its terms,
        or None if the tags are unbalanced, a term is empty or a term does not start and
        end at the edges of words.
    """
    text = ""
    spans = []
    end = 0
    for match in TAG.finditer(line):
        term = match.group(1).strip()
        if not term:
            return None
        text += line[end : match.start()]
        spans.append((len(text), len(text) + len(term)))
        text += term
        end = match.end()
    text += line[end:]
    if "<tag>" in text.lower() or "</tag>" in text.lower():
        return None
    words = [m.span() for m in PIECE.finditer(text)]
    starts = {start for start, _ in words}
    ends = {stop for _, stop in words}
    if any(start not in starts or stop not in ends for start, stop in spans):
        return None
    return text, spans


def make_prompt(method: str, seeds: list[Example], domain: str, k: int) -> str:
    """
    Write a prompt of an LLM method.

    Args:
        method (str): "term", "context" or "combined".
        seeds (List[Example]): The gold sentences, in the order to show them.
        domain (str): The domain.
        k (int): The number of sentences to ask for.

    Returns:
        str: The prompt.
    """
    terms = {text[a:b].lower(): text[a:b] for text, spans in seeds for a, b in spans}
    return PROMPTS[method].format(
        domain=domain,
        k=k,
        sentences="\n".join(tag(text, spans) for text, spans in seeds),
        terms=", ".join(terms.values()),
    )


def parse_sentences(
    answer: str, method: str, terms: list[str] | None = None
) -> list[Example]:
    """
    Read the sentences of a model's answer, one per line.

    Args:
        answer (str): The answer.
        method (str): "term", "context" or "combined".
        terms (Optional[List[str]]): For "context", the terms of the seeds, which are
            looked for in each sentence.

    Returns:
        List[Example]: The sentences with terms, each with the offsets of its terms.
    """
    examples = []
    for line in answer.splitlines():
        line = BULLET.sub("", line.strip()).strip().strip(QUOTES).strip()
        example = untag(line) if line else None
        if example is None:
            continue
        text, spans = example
        if method == "context":
            sentence = Sentence(text, [m.span() for m in PIECE.finditer(text)])
            spans = [
                (sentence.words[first][0], sentence.words[last - 1][1])
                for term in terms or ()
                for first, last in find_term(sentence, term)
            ]
            spans = outermost(spans)
        if spans:
            examples.append((text, spans))
    return examples


def generate(
    client: LLMClient,
    method: str,
    seeds: list[Example],
    domain: str,
    n: int,
    per_prompt: int,
    rng: np.random.Generator,
    seen: set[str],
) -> list[Example]:
    """
    Ask the model for new sentences until there are `n`, or until it has been asked
    three times as often as `n` needs.

    Args:
        client (LLMClient): The model.
        method (str): "term", "context" or "combined".
        seeds (List[Example]): The gold sentences to show.
        domain (str): The domain.
        n (int): The number of sentences wanted.
        per_prompt (int): The number of sentences each prompt asks for.
        rng (np.random.Generator): The random number generator that shuffles the seeds.
        seen (Set[str]): The sentences, in lower case, that are not new; updated.

    Returns:
        List[Example]: The new sentences.
    """
    if not seeds:
        return []
    terms = list({text[a:b].lower() for text, spans in seeds for a, b in spans})
    synthetic: list[Example] = []
    for _ in range(3 * math.ceil(n / per_prompt)):
        if len(synthetic) >= n:
            break
        order = rng.permutation(len(seeds))
        k = min(per_prompt, n - len(synthetic))
        prompt = make_prompt(method, [seeds[i] for i in order], domain, k)
        for text, spans in parse_sentences(client.complete(prompt), method, terms)[:k]:
            if text.lower() not in seen:
                seen.add(text.lower())
                synthetic.append((text, spans))
    return synthetic


def replace_terms(
    seeds: list[Example],
    lexicon: dict[str, list[str]],
    n: int,
    rng: np.random.Generator,
    seen: set[str],
) -> list[Example]:
    """
    Write new sentences by replacing each term of the seeds that is in the lexicon with
    one of its replacements, as in Wiki-TermDA (Banerjee, 2025, §5.1.2).

    The seeds are taken in turn, in a random order, until there are `n` sentences or a
    round of them adds none. The replacements of a term are drawn at random without
    replacement, and drawn again once all have been used.

    Args:
        seeds (List[Example]): The gold sentences.
        lexicon (Dict[str, List[str]]): The replacements of each term, in lower case.
        n (int): The number of sentences wanted.
        rng (np.random.Generator): The random number generator.
        seen (Set[str]): The sentences, in lower case, that are not new; updated.

    Returns:
        List[Example]: The new sentences.
    """
    queues: dict[str, list[str]] = {}

    def draw(term: str) -> str:
        if not queues.get(term):
            queues[term] = list(rng.permutation(lexicon[term]))
        return queues[term].pop()

    synthetic: list[Example] = []
    while len(synthetic) < n:
        added = False
        for i in rng.permutation(len(seeds)):
            text, spans = seeds[i]
            if not any(lexicon.get(text[a:b].lower()) for a, b in spans):
                continue
            new = ""
            new_spans = []
            end = 0
            for start, stop in spans:
                term = text[start:stop].lower()
                replacement = draw(term) if lexicon.get(term) else text[start:stop]
                new += text[end:start]
                new_spans.append((len(new), len(new) + len(replacement)))
                new += replacement
                end = stop
            new += text[end:]
            if new.lower() not in seen:
                seen.add(new.lower())
                synthetic.append((new, new_spans))
                added = True
                if len(synthetic) == n:
                    break
        if not added:
            break
    return synthetic
