import re
import sys
import unicodedata
from collections.abc import Iterator

from .stopwords import is_candidate


def _marks() -> str:
    """A regular expression character class of the Unicode combining marks."""
    ranges = []
    start = None
    for code in range(sys.maxunicode + 2):
        is_mark = code <= sys.maxunicode and unicodedata.category(chr(code))[0] == "M"
        if is_mark and start is None:
            start = code
        elif not is_mark and start is not None:
            ranges.append(f"\\U{start:08x}-\\U{code - 1:08x}")
            start = None
    return "[" + "".join(ranges) + "]"


MARKS = _marks()
"""
A character class of the Unicode combining marks, such as the vowel signs and virama of
Devanagari. Python's `\\w` does not match them, but they are part of words: `शिक्षा` is
`श`, `ि`, `क`, `्`, `ष` and `ा`.
"""

WORD = re.compile(rf"(?:\w|{MARKS})+")
"""A word of plain text: a run of letters, digits, underscores and combining marks."""


def split_words(text: str) -> list[str]:
    """
    Split plain text into lower-case words.

    Args:
        text (str): The text.

    Returns:
        List[str]: The runs of letters, digits, underscores and combining marks.
    """
    return WORD.findall(text.lower())


WORD_PATTERNS = {
    "alphanumeric": r".*[^\W_].*",
    "atr4s": rf"(?:[^\W_]|-|{MARKS}){{3,}}",
}
"""
Named word patterns for `word_pattern`, as regular expressions that every word of a
candidate must fully match.

`alphanumeric` matches words with at least one letter or digit, so candidates such as
"95 % ci" are dropped. `atr4s` is the noise filter of ATR4S (Astrakhantsev, 2018): words
of at least three characters, each a letter, a digit, a hyphen or a combining mark.
"""


def compile_word_pattern(pattern: str) -> re.Pattern[str]:
    """
    Compile a word pattern.

    Args:
        pattern (str): A regular expression, or the name of one of `WORD_PATTERNS`.

    Returns:
        re.Pattern: The compiled pattern, to be matched against whole words.

    Raises:
        ValueError: If the pattern is not a valid regular expression.
    """
    pattern = WORD_PATTERNS.get(pattern, pattern)
    try:
        return re.compile(pattern)
    except re.error as e:
        raise ValueError(f"Invalid word pattern {pattern!r}: {e}") from e


def candidate_spans(
    words: list[str],
    n: int,
    stopwords: frozenset[str],
    word_pattern: re.Pattern[str] | None = None,
    tags: list[str] | None = None,
    pos_pattern: re.Pattern[str] | None = None,
) -> Iterator[tuple[int, int]]:
    """
    Find the occurrences of candidates in a sequence of words.

    An occurrence is a sequence of up to `n` words that neither starts nor ends with a
    stopword, whose words all fully match `word_pattern`, and whose part-of-speech tags
    fully match `pos_pattern`.

    Args:
        words (List[str]): The words of a document.
        n (int): The maximum number of words of a candidate.
        stopwords (FrozenSet[str]): Lower-case stopwords.
        word_pattern (Optional[re.Pattern]): A pattern from `compile_word_pattern`, or
            None to allow every word.
        tags (Optional[List[str]]): The part-of-speech tag of each word, needed if
            `pos_pattern` is given.
        pos_pattern (Optional[re.Pattern]): A pattern from
            `termxtract.pos.compile_pos_pattern`, or None to allow every tag sequence.

    Yields:
        Tuple[int, int]: The start and end (exclusive) word index of each occurrence.
    """
    allowed = [word_pattern is None or bool(word_pattern.fullmatch(w)) for w in words]
    boundary = [is_candidate(w, stopwords) for w in words]
    for i in range(len(words)):
        if not boundary[i]:
            continue
        for j in range(i + 1, min(i + n, len(words)) + 1):
            if not allowed[j - 1]:
                # Every longer sequence from i contains this word too
                break
            if not boundary[j - 1]:
                continue
            if pos_pattern is not None:
                assert tags is not None
                if not pos_pattern.fullmatch("".join(f"<{t}>" for t in tags[i:j])):
                    continue
            yield i, j
