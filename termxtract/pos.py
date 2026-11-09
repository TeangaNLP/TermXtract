import re

from teanga import Corpus

from .layers import pos_tags, words_with_offsets
from .utils import ATEResults

POS_PATTERNS = {
    "noun_phrase": (
        "<ADJ|NOUN|PROPN>*<NOUN|PROPN>(<ADP><DET>?<ADJ|NOUN|PROPN>*<NOUN|PROPN>)?"
    ),
}
"""
Named part-of-speech patterns, using Universal Dependencies tags.

`noun_phrase` matches nouns with optional adjectives or nouns before them, such as
`photosynthesis` or `chemical energy`, optionally followed by a preposition and a second
such phrase, such as `degree of freedom`.
"""


def compile_pos_pattern(pattern: str) -> re.Pattern[str]:
    """
    Compile a part-of-speech pattern into a regular expression over tag sequences.

    Patterns use the syntax of NLTK's `RegexpParser`: each tag is written in angle
    brackets and combined with regular-expression operators, as in
    `<ADJ|NOUN>*<NOUN>`. Inside angle brackets, `|` separates alternative tags and `.`
    matches any character of a tag, so `<N.*>` matches `NOUN` and `NUM`. Whitespace is
    ignored.

    Args:
        pattern (str): A pattern, or the name of one of `POS_PATTERNS`.

    Returns:
        re.Pattern: A regular expression that fully matches the tag sequences, written
        as `<TAG1><TAG2>...`, that the pattern allows.

    Raises:
        ValueError: If the pattern is not a valid regular expression.
    """
    pattern = POS_PATTERNS.get(pattern, pattern)

    def tag(match: re.Match[str]) -> str:
        return "(?:<(?:" + match.group(1).replace(".", "[^<>]") + ")>)"

    regex = re.sub(r"<([^<>]*)>", tag, re.sub(r"\s+", "", pattern))
    try:
        return re.compile(regex)
    except re.error as e:
        raise ValueError(f"Invalid part-of-speech pattern {pattern!r}: {e}") from e


def filter_by_pos(
    results: ATEResults,
    corpus: Corpus,
    pattern: re.Pattern[str],
    pos_layer: str | None = None,
    words_layer: str | None = None,
    lemma_layer: str | None = None,
) -> ATEResults:
    """
    Keep the terms that match a part-of-speech pattern in at least one occurrence.

    Terms are matched to the word sequences of the corpus case-insensitively. Scores
    are unchanged, and offsets are kept only for the remaining terms.

    Args:
        results (ATEResults): The results to filter.
        corpus (Corpus): The Teanga corpus the results were extracted from.
        pattern (re.Pattern): A pattern from `compile_pos_pattern`.
        pos_layer (Optional[str]): The part-of-speech layer, or None for `upos` or,
            failing that, `pos`.
        words_layer (Optional[str]): The words layer, or None for `words` or, failing
            that, `tokens`.
        lemma_layer (Optional[str]): The lemma layer the terms were extracted with, if
            any, so that terms are matched to the lemmas of the words.

    Returns:
        ATEResults: The filtered results.
    """
    terms = {t["term"] for doc in results.terms for t in doc["terms"]}
    max_length = max((len(term.split()) for term in terms), default=0)

    matching = set()
    for doc_id in corpus.doc_ids:
        doc = corpus.doc_by_id(doc_id)
        words = [
            word.lower()
            for _, _, word in words_with_offsets(doc, words_layer, lemma_layer)
        ]
        tags = [f"<{tag}>" for tag in pos_tags(doc, pos_layer, words_layer)]
        for i in range(len(words)):
            for k in range(1, min(max_length, len(words) - i) + 1):
                if pattern.fullmatch("".join(tags[i : i + k])):
                    matching.add(" ".join(words[i : i + k]))

    kept = {term for term in terms if " ".join(term.lower().split()) in matching}
    return ATEResults(
        corpus=results.corpus,
        terms=[
            {**doc, "terms": [t for t in doc["terms"] if t["term"] in kept]}
            for doc in results.terms
        ],
        offsets=None
        if results.offsets is None
        else [
            {**doc, "offsets": [o for o in doc["offsets"] if o["term"] in kept]}
            for doc in results.offsets
        ],
    )
