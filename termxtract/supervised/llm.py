import os
import re
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from teanga import Corpus

from ..candidates import WORD
from ..datasets.base import GOLD_LAYER
from ..embeddings import Encoder
from ..evaluation import normalize_term
from ..layers import find_layer
from ..llm import LLMClient, OpenAIClient
from ..stopwords import Stopwords, load_stopwords
from ..utils import ATEResults, rank_terms
from .sentences import teanga_lemmas, teanga_sentences

INSTRUCTIONS = (
    "Identify the terms in the following text.",
    "Extract the terms from the following text.",
    "Find the domain-specific terms in the following text.",
    (
        "Identify the words or phrases that are relevant to the underlying domain of "
        "the input text."
    ),
)
"""The task instructions of Banerjee et al. (2024)."""

MARKERS = (("Input:", "Output:"), ("Text:", "Terminology:"))
"""The markers of the input text and of its terms of Banerjee et al. (2024)."""

TEMPLATES = tuple(
    (instruction, *markers) for instruction in INSTRUCTIONS for markers in MARKERS
)
"""
The eight prompt templates of Banerjee et al. (2024, Table 4.3 of Banerjee, 2025), as
(instruction, input marker, output marker): template 1 is `TEMPLATES[0]`. Each
instruction is paired with each pair of markers.
"""

SELECTIONS = ("knn", "random")
"""The ways of choosing the demonstrations of each prompt."""

NO_TERMS = "None"
"""The answer of a demonstration without terms."""

SENTENCE = re.compile(r"\S(?:.*?\S)?(?=\s*$|(?<=[.!?])\s)", re.DOTALL)
"""A sentence of plain text: up to a ".", "!" or "?" followed by whitespace."""

PIECE = re.compile(rf"{WORD.pattern}|[^\w\s]")
"""A word or punctuation mark of a term written by the model."""

BULLET = re.compile(r"^(?:[-*•]+|\d+[.)])\s+")
"""A list marker before a term."""

STRIP = " \t\"'`“”‘’.:;!?"
"""Characters stripped from both ends of a term written by the model."""

ABBREVIATION = re.compile(r"(.+?)\s+\((.+)\)")
"""A term followed by its abbreviation in brackets, as in "hazard ratio (HR)"."""


@dataclass
class Sentence:
    """A sentence, with the offsets of its words within it."""

    text: str
    words: list[tuple[int, int]]
    lemmas: list[str] | None = None
    offset: int = 0
    """The offset of the sentence in its document."""

    def words_as_written(self) -> list[str]:
        """The words of the sentence."""
        return [self.text[start:end] for start, end in self.words]

    def term(self, first: int, last: int) -> str:
        """The term made of words `first` to `last` - 1, by their lemmas if any."""
        if self.lemmas is not None:
            return " ".join(self.lemmas[first:last]).lower()
        return normalize_term(self.text[self.words[first][0] : self.words[last - 1][1]])


def string_sentences(text: str) -> list[Sentence]:
    """
    Split plain text into sentences, after each ".", "!" or "?" followed by whitespace.

    Args:
        text (str): The text.

    Returns:
        List[Sentence]: The sentences that have words.
    """
    sentences = []
    for part in SENTENCE.finditer(text):
        words = [m.span() for m in WORD.finditer(part.group())]
        if words:
            sentences.append(Sentence(part.group(), words, offset=part.start()))
    return sentences


def make_prompt(
    template: tuple[str, str, str],
    demonstrations: Iterable[tuple[str, list[str]]],
    text: str,
) -> str:
    """
    Write the prompt for a sentence: the instruction, each demonstration with its terms,
    and the sentence.

    Args:
        template (Tuple[str, str, str]): The instruction, the input marker and the
            output marker, as in `TEMPLATES`.
        demonstrations (Iterable[Tuple[str, List[str]]]): Example sentences, each with
            its terms.
        text (str): The sentence to find the terms of.

    Returns:
        str: The prompt.
    """
    instruction, input_marker, output_marker = template
    blocks = [instruction]
    for example, terms in demonstrations:
        answer = ", ".join(terms) if terms else NO_TERMS
        blocks.append(f"{input_marker} {example}\n{output_marker} {answer}")
    blocks.append(f"{input_marker} {text}\n{output_marker}")
    return "\n\n".join(blocks)


def parse_answer(answer: str, template: tuple[str, str, str]) -> list[str]:
    """
    Read the terms from a model's answer.

    The terms are split at commas or, if there are none, at newlines and semicolons.
    Anything after the input marker, where the model starts a new example, is ignored,
    as is "None". A term followed by its abbreviation in brackets, as in "hazard ratio
    (HR)", gives the term and the abbreviation.

    Args:
        answer (str): The answer.
        template (Tuple[str, str, str]): The template of the prompt.

    Returns:
        List[str]: The terms, as written, without duplicates.
    """
    _, input_marker, output_marker = template
    answer = answer.split(input_marker)[0].strip()
    answer = answer.removeprefix(output_marker)
    separators = r"[,\n]" if "," in answer else r"[;\n]"
    terms: dict[str, str] = {}
    for part in re.split(separators, answer):
        term = unbracket(BULLET.sub("", part.strip()).strip(STRIP))
        match = ABBREVIATION.fullmatch(term)
        for found in match.groups() if match else (term,):
            if found and found.lower() != NO_TERMS.lower():
                terms.setdefault(found.lower(), found)
    return list(terms.values())


def unbracket(term: str) -> str:
    """Strip a bracket from either end of a term if it has no partner, or a pair of
    brackets around the whole term."""
    while True:
        if term[:1] == "(" and term[-1:] == ")" and term.count("(") == 1:
            term = term[1:-1].strip()
        elif term[:1] == "(" and term.count("(") > term.count(")"):
            term = term[1:].strip()
        elif term[-1:] == ")" and term.count(")") > term.count("("):
            term = term[:-1].strip()
        else:
            return term


def find_term(sentence: Sentence, term: str) -> list[tuple[int, int]]:
    """
    Find a term written by the model in a sentence, ignoring case and spacing.

    Args:
        sentence (Sentence): The sentence.
        term (str): The term.

    Returns:
        List[Tuple[int, int]]: The index of the first word and of the word after the
        last of each occurrence that starts and ends at the edges of words.
    """
    pieces = PIECE.findall(term)
    if not pieces:
        return []
    pattern = re.escape(pieces[0])
    for previous, piece in pairwise(pieces):
        # Words must be apart, as in "heart failure"; punctuation need not be
        apart = WORD.fullmatch(previous) and WORD.fullmatch(piece)
        pattern += (r"\s+" if apart else r"\s*") + re.escape(piece)
    starts = {start: i for i, (start, _) in enumerate(sentence.words)}
    ends = {end: i + 1 for i, (_, end) in enumerate(sentence.words)}
    return [
        (starts[m.start()], ends[m.end()])
        for m in re.finditer(pattern, sentence.text, re.IGNORECASE)
        if m.start() in starts and m.end() in ends
    ]


class LLMTermExtractor:
    """
    Term extraction by in-context learning with a large language model, as in Banerjee
    et al. (2024).

    Each sentence is sent to the model in a prompt with `k` training sentences and their
    terms, the demonstrations, and the model answers with the terms of the sentence.
    The demonstrations are the training sentences most similar to it (Liu et al., 2022)
    or a random sample of them.
    """

    def __init__(
        self,
        client: LLMClient | str | None = None,
        k: int = 5,
        template: int | tuple[str, str, str] = 5,
        selection: str = "knn",
        encoder: Encoder | None = None,
        n: int = 10,
        random_state: int | None = None,
        n_jobs: int | None = None,
        stopwords: Stopwords = "english",
        words_layer: str | None = None,
        lemma_layer: str | None = None,
        terms_layer: str = GOLD_LAYER,
        sentences_layer: str | None = None,
    ):
        """
        Initialize the extractor.

        Args:
            client (Union[LLMClient, str]): The model: an `LLMClient`, such as an
                `OpenAIClient`, or the name of a model for an `OpenAIClient` configured
                by the `OPENAI_BASE_URL` and `OPENAI_API_KEY` environment variables.
            k (int): The number of demonstrations in each prompt; 0 asks the model
                without examples.
            template (Union[int, Tuple[str, str, str]]): The number of a template of
                `TEMPLATES`, from 1 to 8, or an (instruction, input marker, output
                marker) tuple. Template 5, "Find the domain-specific terms in the
                following text." with `Input:` and `Output:`, did best in Banerjee et
                al. (2024).
            selection (str): "knn" for the training sentences most similar to each
                sentence, by the cosine of their embeddings, with the most similar last,
                or "random" for a random sample of them.
            encoder (Optional[Encoder]): The sentence encoder of "knn", or None for
                `Encoder()`, which needs the 'transformers' extra.
            n (int): Drop terms of more than this many words.
            random_state (Optional[int]): Seed of the "random" selection.
            n_jobs (Optional[int]): The number of prompts sent at once, -1 for one per
                CPU, or None for one.
            stopwords (Union[str, Iterable[str], None]): Drop terms that start or end
                with a stopword, as for `TermExtractor`.
            words_layer (Optional[str]): For Teanga input, the span layer holding the
                words, or None for `words` or, failing that, `tokens`.
            lemma_layer (Optional[str]): For Teanga input, a sequence layer holding the
                lemma of each word, to give each term by its lemmas and match labels to
                them, or None to give terms as written.
            terms_layer (str): For Teanga input without labels, the span layer of the
                gold terms to take the demonstrations from.
            sentences_layer (Optional[str]): For Teanga input, the span layer of the
                sentences, or None for `sentences` or, if there is none, splitting after
                each word that is ".", "!" or "?".

        Raises:
            ValueError: If there is no client, or the template, `k` or the selection
                is invalid.
        """
        if client is None:
            raise ValueError("The 'llm' method needs a client, or the name of a model.")
        if isinstance(template, int):
            if not 1 <= template <= len(TEMPLATES):
                raise ValueError(f"template must be from 1 to {len(TEMPLATES)}.")
            template = TEMPLATES[template - 1]
        elif len(template) != 3:
            raise ValueError(
                "template must be a number or an (instruction, input marker, output "
                "marker) tuple."
            )
        if k < 0:
            raise ValueError("k must be at least 0.")
        if selection not in SELECTIONS:
            raise ValueError(
                f"Unknown selection: {selection!r}. Use one of {', '.join(SELECTIONS)}."
            )
        self.client = OpenAIClient(client) if isinstance(client, str) else client
        self.k = k
        self.template = template
        self.selection = selection
        self.encoder = encoder
        self.n = n
        self.random_state = random_state
        self.n_jobs = os.cpu_count() if n_jobs == -1 else n_jobs
        self.stopwords = load_stopwords(stopwords)
        self.words_layer = words_layer
        self.lemma_layer = lemma_layer
        self.terms_layer = terms_layer
        self.sentences_layer = sentences_layer
        self.demonstrations: list[tuple[str, list[str]]] | None = None
        self.vectors: np.ndarray | None = None

    def fit_strings(self, corpus: list[str], labels: dict[str, int]) -> None:
        """
        Take the demonstrations from a list of strings, each sentence with the terms
        labelled 1 that it contains.

        Args:
            corpus (List[str]): List of text documents.
            labels (Dict[str, int]): Dictionary with terms as keys and binary labels as
                values.
        """
        self._fit(
            [s for doc in corpus for s in string_sentences(doc)],
            self._labelled(labels),
        )

    def fit_teanga(self, corpus: Corpus, labels: dict[str, int] | None = None) -> None:
        """
        Take the demonstrations from a Teanga corpus, each sentence with its gold terms
        or with the terms labelled 1 that it contains.

        Args:
            corpus (Corpus): Teanga corpus object.
            labels (Optional[Dict[str, int]]): Dictionary with terms as keys and binary
                labels as values, or None for the gold terms of `terms_layer`.
        """
        documents = self._teanga_sentences(corpus)
        sentences = [s for doc in documents for s in doc]
        if labels is not None:
            self._fit(sentences, self._labelled(labels))
            return
        layer = find_layer(corpus.meta, self.terms_layer, (), "terms_layer")
        gold = []
        for doc_id, doc in zip(corpus.doc_ids, documents):
            document = corpus.doc_by_id(doc_id)
            spans = sorted(
                (span[0], span[1]) for span in document[layer].indexes("text")
            )
            for sentence in doc:
                end = sentence.offset + len(sentence.text)
                terms = {
                    document.text[start:stop].lower(): document.text[start:stop]
                    for start, stop in spans
                    if sentence.offset <= start < end
                }
                gold.append((sentence.text, list(terms.values())))
        self._set_demonstrations(sentences, gold)

    def predict_strings(self, corpus: list[str]) -> ATEResults:
        """
        Extract the terms of a list of strings.

        Args:
            corpus (List[str]): List of text documents.

        Returns:
            ATEResults: The terms found in each document, each scoring 1.0.
        """
        return self._predict(
            corpus,
            [f"doc_{i}" for i in range(len(corpus))],
            [string_sentences(doc) for doc in corpus],
            offsets=False,
        )

    def predict_teanga(self, corpus: Corpus) -> ATEResults:
        """
        Extract the terms of a Teanga corpus.

        Args:
            corpus (Corpus): Teanga corpus object.

        Returns:
            ATEResults: The terms found in each document, each scoring 1.0, with the
            offsets of each occurrence the model found.
        """
        return self._predict(
            corpus, list(corpus.doc_ids), self._teanga_sentences(corpus), offsets=True
        )

    def prompts(self, sentences: list[Sentence]) -> list[str]:
        """
        Write the prompt of each sentence, with its demonstrations.

        A training sentence is never a demonstration for itself, so that the terms of
        the training corpus can be extracted.

        Args:
            sentences (List[Sentence]): The sentences.

        Returns:
            List[str]: The prompt of each sentence.

        Raises:
            RuntimeError: If the extractor has not been trained.
        """
        if self.demonstrations is None:
            raise RuntimeError("The extractor must be trained before prediction.")
        demonstrations = self.demonstrations
        texts = np.array([text for text, _ in demonstrations], dtype=object)
        knn = self.selection == "knn" and self.k > 0 and len(texts) > 0
        if knn and sentences:
            assert self.encoder is not None and self.vectors is not None
            similarity = (
                self.encoder.encode([s.words_as_written() for s in sentences])
                @ self.vectors.T
            )
        rng = np.random.default_rng(self.random_state)
        prompts = []
        for i, sentence in enumerate(sentences):
            # A sentence is no example for itself
            itself = texts == sentence.text
            if knn:
                scores = np.where(itself, -np.inf, similarity[i])
                ranked = np.argsort(-scores, kind="stable")
                k = min(self.k, int((~itself).sum()))
                # The most similar is next to the sentence
                chosen = ranked[:k][::-1]
            else:
                allowed = np.flatnonzero(~itself)
                chosen = rng.choice(allowed, min(self.k, len(allowed)), replace=False)
            prompts.append(
                make_prompt(
                    self.template, (demonstrations[j] for j in chosen), sentence.text
                )
            )
        return prompts

    def _teanga_sentences(self, corpus: Corpus) -> list[list[Sentence]]:
        """The sentences of each document of a Teanga corpus."""
        documents = teanga_sentences(corpus, self.words_layer, self.sentences_layer)
        lemmas = teanga_lemmas(corpus, documents, self.lemma_layer, self.words_layer)
        sentences = []
        for k, (doc_id, doc) in enumerate(zip(corpus.doc_ids, documents)):
            text = corpus.doc_by_id(doc_id).text
            doc_sentences = []
            for s, (_, spans) in enumerate(doc):
                start = spans[0][0]
                doc_sentences.append(
                    Sentence(
                        text[start : spans[-1][1]],
                        [(a - start, b - start) for a, b in spans],
                        None if lemmas is None else lemmas[k][s],
                        start,
                    )
                )
            sentences.append(doc_sentences)
        return sentences

    def _labelled(self, labels: dict[str, int]) -> set[str]:
        """The terms labelled 1, normalised."""
        return {normalize_term(term) for term, label in labels.items() if label == 1}

    def _fit(self, sentences: list[Sentence], terms: set[str]) -> None:
        """Take the demonstrations from sentences, each with the terms it contains."""
        gold = []
        for sentence in sentences:
            found: dict[str, str] = {}
            for first in range(len(sentence.words)):
                for last in range(
                    first + 1, min(first + self.n, len(sentence.words)) + 1
                ):
                    if sentence.term(first, last) in terms:
                        text = sentence.text[
                            sentence.words[first][0] : sentence.words[last - 1][1]
                        ]
                        found.setdefault(text.lower(), text)
            gold.append((sentence.text, list(found.values())))
        self._set_demonstrations(sentences, gold)

    def _set_demonstrations(
        self, sentences: list[Sentence], demonstrations: list[tuple[str, list[str]]]
    ) -> None:
        """Keep the demonstrations and, for "knn", the embeddings of their sentences."""
        self.demonstrations = demonstrations
        if self.selection == "knn" and self.k > 0:
            if self.encoder is None:
                self.encoder = Encoder()
            self.vectors = self.encoder.encode(
                [s.words_as_written() for s in sentences]
            )

    def _predict(
        self,
        corpus: Corpus | list[str],
        doc_ids: list[str],
        documents: list[list[Sentence]],
        offsets: bool,
    ) -> ATEResults:
        """Ask the model for the terms of each sentence of each document."""
        sentences = [s for doc in documents for s in doc]
        prompts = self.prompts(sentences)
        if self.n_jobs is None or self.n_jobs == 1:
            answers = list(map(self.client.complete, prompts))
        else:
            with ThreadPoolExecutor(self.n_jobs) as pool:
                answers = list(pool.map(self.client.complete, prompts))
        replies = iter(answers)
        terms_by_doc = []
        offsets_by_doc = []
        for doc_id, doc in zip(doc_ids, documents):
            found: list[tuple[str, int, int]] = []
            for sentence in doc:
                for written in parse_answer(next(replies), self.template):
                    for first, last in find_term(sentence, written):
                        term = sentence.term(first, last)
                        if last - first <= self.n and any(c.isalpha() for c in term):
                            found.append(
                                (
                                    term,
                                    sentence.offset + sentence.words[first][0],
                                    sentence.offset + sentence.words[last - 1][1],
                                )
                            )
            # The model lists terms rather than scoring them, so every term scores 1.0
            ranked = rank_terms(
                dict.fromkeys((term for term, _, _ in found), 1.0),
                stopwords=self.stopwords,
            )
            terms_by_doc.append({"doc_id": doc_id, "terms": ranked})
            kept = {term["term"] for term in ranked}
            offsets_by_doc.append(
                {
                    "doc_id": doc_id,
                    "offsets": [
                        {"term": term, "offset": (start, end)}
                        for term, start, end in sorted(set(found), key=lambda t: t[1:])
                        if term in kept
                    ],
                }
            )
        return ATEResults(
            corpus=corpus,
            terms=terms_by_doc,
            offsets=offsets_by_doc if offsets else None,
        )
