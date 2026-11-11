"""
The Irish Wikipedia term extraction gold standard: 11 Irish Wikipedia articles with
their terms annotated.
"""

import re
from pathlib import Path

from teanga import Corpus

from .base import GOLD_LAYER, SENTENCES_LAYER, DatasetInfo, Resource, register

# The tokens that end a sentence
_SENTENCE_ENDS = {".", "?", "!", "..."}

# A term needs a letter or a digit: two annotations are a lone quotation mark
_WORD_CHARACTER = re.compile(r"\w")


def load_irish_wikipedia(path: str | Path, *, split: str | None = None) -> Corpus:
    """
    Load the Irish Wikipedia gold standard from its tagged file.

    Each line of the file is an article, as tokens with B, I or O tags. Each article is
    a document, with these layers:

    - `text`: the tokens, joined by spaces. The first tokens are the article's title,
      which runs into its first sentence, as in the file.
    - `tokens` and `sentences`. A sentence ends after `.`, `?`, `!` or `...`.
    - `terms` (`GOLD_LAYER`): the terms, from the B and I tags, which do not nest.
      The two terms that are a lone quotation mark are left out.

    Args:
        path (Union[str, Path]): The tagged file.
        split (Optional[str]): Not used: the corpus has no splits.

    Returns:
        Corpus: The articles, in the order of the file.
    """
    corpus = Corpus()
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("tokens", layer_type="span", base="text")
    corpus.add_layer_meta(SENTENCES_LAYER, layer_type="span", base="text")
    corpus.add_layer_meta(GOLD_LAYER, layer_type="span", base="text")
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        # Some tokens contain "_", so the tag follows the last one
        tagged = [token.rsplit("_", 1) for token in line.split()]
        if not tagged:
            continue
        words: list[str] = []
        tokens: list[tuple[int, int]] = []
        sentences: list[tuple[int, int]] = []
        terms: list[tuple[int, int]] = []
        start = sentence = 0
        for word, tag in tagged:
            tokens.append((start, start + len(word)))
            words.append(word)
            if tag == "B" or (tag == "I" and (not terms or terms[-1][1] != start - 1)):
                terms.append((start, start + len(word)))
            elif tag == "I":
                terms[-1] = (terms[-1][0], start + len(word))
            if word in _SENTENCE_ENDS:
                sentences.append((sentence, start + len(word)))
                sentence = start + len(word) + 1
            start += len(word) + 1
        text = " ".join(words)
        if sentence < len(text):
            sentences.append((sentence, len(text)))
        doc = corpus.add_doc(text=text)
        doc.tokens = tokens
        doc[SENTENCES_LAYER] = sentences
        doc[GOLD_LAYER] = [
            (s, e) for s, e in terms if _WORD_CHARACTER.search(text[s:e])
        ]
    return corpus


IRISH_WIKIPEDIA = register(
    DatasetInfo(
        name="irish-wikipedia",
        description=(
            "The Irish Wikipedia gold standard: 11 Irish Wikipedia articles with their "
            "terms annotated."
        ),
        languages=("ga",),
        domains=("general",),
        license="CC BY-SA 4.0, as the Irish Wikipedia texts",
        citation=(
            "McCrae, J. P., & Doyle, A. (2019). Adapting term recognition to an "
            "under-resourced language: the case of Irish. In Proceedings of the Celtic "
            "Language Technology Workshop (pp. 48–57). European Association for "
            "Machine Translation."
        ),
        homepage="https://doi.org/10.5281/zenodo.24132067",
        resources=(
            Resource(
                "https://zenodo.org/records/24132067/files/"
                "irish-wikipedia-terms.txt?download=1",
                "9ecb3c33f91e29a3a48dc56f11c5ff61522f76933ed33391b8bcfe96a2e1d68c",
                "irish-wikipedia-terms.txt",
            ),
        ),
        loader=load_irish_wikipedia,
    )
)
