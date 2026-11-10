import json
from collections.abc import Iterable
from importlib.resources import files

# NLTK's English stopword list (nltk_data, corpora/stopwords/english), which is based on
# the Snowball project's list (BSD licence): https://github.com/nltk/nltk_data
ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "ain",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren",
        "aren't",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "couldn",
        "couldn't",
        "d",
        "did",
        "didn",
        "didn't",
        "do",
        "does",
        "doesn",
        "doesn't",
        "doing",
        "don",
        "don't",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "hadn",
        "hadn't",
        "has",
        "hasn",
        "hasn't",
        "have",
        "haven",
        "haven't",
        "having",
        "he",
        "he'd",
        "he'll",
        "her",
        "here",
        "hers",
        "herself",
        "he's",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "i'd",
        "if",
        "i'll",
        "i'm",
        "in",
        "into",
        "is",
        "isn",
        "isn't",
        "it",
        "it'd",
        "it'll",
        "it's",
        "its",
        "itself",
        "i've",
        "just",
        "ll",
        "m",
        "ma",
        "me",
        "mightn",
        "mightn't",
        "more",
        "most",
        "mustn",
        "mustn't",
        "my",
        "myself",
        "needn",
        "needn't",
        "no",
        "nor",
        "not",
        "now",
        "o",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "re",
        "s",
        "same",
        "shan",
        "shan't",
        "she",
        "she'd",
        "she'll",
        "she's",
        "should",
        "shouldn",
        "shouldn't",
        "should've",
        "so",
        "some",
        "such",
        "t",
        "than",
        "that",
        "that'll",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "they'd",
        "they'll",
        "they're",
        "they've",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "ve",
        "very",
        "was",
        "wasn",
        "wasn't",
        "we",
        "we'd",
        "we'll",
        "we're",
        "were",
        "weren",
        "weren't",
        "we've",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "won",
        "won't",
        "wouldn",
        "wouldn't",
        "y",
        "you",
        "you'd",
        "you'll",
        "your",
        "you're",
        "yours",
        "yourself",
        "yourselves",
        "you've",
    }
)


def _stopwords_iso() -> dict[str, frozenset[str]]:
    """The lists of stopwords-iso, by ISO 639-1 code."""
    path = files("termxtract").joinpath("data").joinpath("stopwords-iso.json")
    data = path.read_text(encoding="utf-8")
    return {code: frozenset(words) for code, words in json.loads(data).items()}


STOPWORDS_ISO = _stopwords_iso()
"""
The stopword lists of 58 languages from stopwords-iso (MIT licence), by ISO 639-1 code,
pinned to commit 6e8e9a2: https://github.com/stopwords-iso/stopwords-iso

The lists are bundled in `termxtract/data/stopwords-iso.json`, with their licence.
"""

STOPWORD_LISTS: dict[str, frozenset[str]] = {
    **STOPWORDS_ISO,
    "en": ENGLISH_STOPWORDS,
    "english": ENGLISH_STOPWORDS,
    "hindi": STOPWORDS_ISO["hi"],
    "slovenian": STOPWORDS_ISO["sl"],
}
"""
The built-in stopword lists, by the name passed as `stopwords`: the ISO 639-1 code of a
language in `STOPWORDS_ISO`, such as `"fr"`, `"ga"` or `"sl"`, or `"english"`,
`"hindi"` or `"slovenian"`.

English, as `"en"` or `"english"`, is NLTK's list of 198 function words, not
stopwords-iso's list of 1,298 words, which includes many content words. The lists of
languages written without spaces between words, such as Chinese (`"zh"`), Japanese
(`"ja"`) and Thai (`"th"`), only match words in a Teanga corpus whose words layer is
already segmented: plain text in these languages is not split into words.
"""

Stopwords = str | Iterable[str] | None


def load_stopwords(stopwords: Stopwords) -> frozenset[str]:
    """
    Resolve a stopwords argument to a set of lower-case words.

    Args:
        stopwords (Union[str, Iterable[str], None]): The name of a built-in list in
            `STOPWORD_LISTS`, such as "english" or an ISO 639-1 code such as "fr", any
            iterable of words for a custom list, or None for no stopwords.

    Returns:
        FrozenSet[str]: The stopwords, in lower case.

    Raises:
        ValueError: If `stopwords` is a string that names no built-in list.
    """
    if stopwords is None:
        return frozenset()
    if isinstance(stopwords, str):
        if stopwords in STOPWORD_LISTS:
            return STOPWORD_LISTS[stopwords]
        raise ValueError(
            f"Unknown stopword list: {stopwords!r}. Use the name of a list in "
            "termxtract.stopwords.STOPWORD_LISTS, such as 'english' or an ISO 639-1 "
            "code such as 'fr', a list of words, or None for no stopword filtering."
        )
    return frozenset(word.lower() for word in stopwords)


def is_candidate(term: str, stopwords: frozenset[str]) -> bool:
    """
    Check that a candidate term neither starts nor ends with a stopword.

    Stopwords inside a multi-word term are allowed, as in "degree of freedom".

    Args:
        term (str): The candidate, with words separated by spaces.
        stopwords (FrozenSet[str]): Lower-case stopwords.

    Returns:
        bool: True if the first and last words of the term are not stopwords.
    """
    words = term.split()
    return (
        bool(words)
        and words[0].lower() not in stopwords
        and words[-1].lower() not in stopwords
    )
