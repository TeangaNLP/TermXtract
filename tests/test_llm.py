import importlib.util
import re
from types import SimpleNamespace

import numpy as np
import pytest
from teanga import Corpus

from termxtract import SupervisedTermExtractor
from termxtract.embeddings import Encoder
from termxtract.supervised.llm import (
    TEMPLATES,
    LLMTermExtractor,
    Sentence,
    find_term,
    make_prompt,
    parse_answer,
    string_sentences,
)

HAS_OPENAI = importlib.util.find_spec("openai") is not None

TEMPLATE = TEMPLATES[0]


class FakeClient:
    """Answers each prompt from a dictionary from sentences to answers."""

    def __init__(self, answers=None):
        self.answers = answers or {}
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        query = prompt.rsplit("\n\n", 1)[1].split("\n")[0].split(" ", 1)[1]
        return self.answers.get(query, "None")


class FakeEncoder(Encoder):
    """Embeds a sentence by the words it shares with a fixed vocabulary."""

    VOCABULARY = ("heart", "failure", "wind", "turbine", "blade")

    def __init__(self):
        self.tokenizer = SimpleNamespace(mask_token_id=0)

    def encode(self, texts, masks=None):
        vectors = np.array(
            [
                [1.0] + [float(w in (t.lower() for t in text)) for w in self.VOCABULARY]
                for text in texts
            ]
        )
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def corpus(texts, terms):
    """A Teanga corpus with words and sentences, and gold terms by their text."""
    corpus = Corpus()
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("words", layer_type="span", base="text")
    corpus.add_layer_meta("lemma", layer_type="seq", base="words", data="string")
    corpus.add_layer_meta("terms", layer_type="span", base="text")
    for text, doc_terms in zip(texts, terms):
        doc = corpus.add_doc(text)
        words = [m.span() for m in re.finditer(r"\w+|[^\w\s]", text)]
        doc.words = words
        # A toy lemmatiser that drops a plural "s"
        doc.lemma = [re.sub(r"(?<=\w\w)s$", "", text[a:b].lower()) for a, b in words]
        doc.terms = sorted(
            m.span() for term in doc_terms for m in re.finditer(re.escape(term), text)
        )
    return corpus


def extractor(client, **params):
    return LLMTermExtractor(client=client, encoder=FakeEncoder(), **params)


def test_make_prompt():
    prompt = make_prompt(
        TEMPLATES[5], [("Heart failure.", ["heart failure"]), ("Hi.", [])], "Wind."
    )
    assert prompt == (
        "Find the domain-specific terms in the following text.\n\n"
        "Text: Heart failure.\nTerminology: heart failure\n\n"
        "Text: Hi.\nTerminology: None\n\n"
        "Text: Wind.\nTerminology:"
    )
    assert len(TEMPLATES) == 8
    assert TEMPLATES[4] == (
        "Find the domain-specific terms in the following text.",
        "Input:",
        "Output:",
    )


def test_parse_answer():
    assert parse_answer("heart failure, LVEF , heart failure", TEMPLATE) == [
        "heart failure",
        "LVEF",
    ]
    assert parse_answer("- heart failure\n2. 'LVEF'\n", TEMPLATE) == [
        "heart failure",
        "LVEF",
    ]
    assert parse_answer("heart failure; LVEF.", TEMPLATE) == ["heart failure", "LVEF"]
    assert parse_answer("Output: heart failure\n\nInput: Wind.", TEMPLATE) == [
        "heart failure"
    ]
    assert parse_answer(
        "hazard ratio (HR), Ca(2+), (LVEF), LVEF), (NYHA", TEMPLATE
    ) == [
        "hazard ratio",
        "HR",
        "Ca(2+)",
        "LVEF",
        "NYHA",
    ]
    assert parse_answer("None", TEMPLATE) == []
    assert parse_answer("", TEMPLATE) == []


def test_find_term():
    [sentence] = string_sentences("Beta-blockers and beta - blockers, not hearts.")
    assert find_term(sentence, "BETA-BLOCKERS") == [(0, 2), (3, 5)]
    # A term must start and end at the edges of words
    assert find_term(sentence, "heart") == []
    assert find_term(sentence, "blockers and beta") == [(1, 4)]
    assert find_term(sentence, "...") == []


def test_string_sentences():
    sentences = string_sentences("Heart failure. Wind turbines!  ")
    assert [(s.text, s.offset) for s in sentences] == [
        ("Heart failure.", 0),
        ("Wind turbines!", 15),
    ]
    assert sentences[1].words_as_written() == ["Wind", "turbines"]


def test_extract_strings():
    client = FakeClient(
        {
            "Acute heart failure is common.": "acute heart failure, heart failure, "
            "the heart, 2010, kidney",
            "We met on Tuesday.": "None",
        }
    )
    train = ["Chronic heart failure was treated. It rained."]
    found = (
        SupervisedTermExtractor(method="llm", client=client, encoder=FakeEncoder(), k=2)
        .fit(train, {"chronic heart failure": 1, "heart failure": 1, "rain": 1})
        .predict(["Acute heart failure is common. We met on Tuesday."])
    )
    # "the heart" starts with a stopword, "2010" is a number, and "kidney" is not in
    # the sentence
    assert [t["term"] for t in found.terms[0]["terms"]] == [
        "acute heart failure",
        "heart failure",
    ]
    assert found.offsets is None
    first = client.prompts[0]
    assert (
        "Input: Chronic heart failure was treated.\n"
        "Output: Chronic heart failure, heart failure" in first
    )
    assert "Input: It rained.\nOutput: None" in first


def test_extract_teanga():
    texts = ["Heart failure patients. Wind turbine blades.", "Turbine blades crack."]
    gold = [["Heart failure", "Wind turbine blades", "turbine blades"], []]
    client = FakeClient(
        {"Turbine blades crack.": "turbine blades", "Wind turbine blades.": "turbine"}
    )
    model = extractor(client, k=1, lemma_layer="lemma")
    model.fit_teanga(corpus(texts, gold))
    assert model.demonstrations == [
        ("Heart failure patients.", ["Heart failure"]),
        ("Wind turbine blades.", ["Wind turbine blades", "turbine blades"]),
        ("Turbine blades crack.", []),
    ]
    found = model.predict_teanga(corpus(texts, [[], []]))
    # Terms are given by their lemmas, with the offsets of what the model found
    assert found.terms[1]["terms"] == [{"term": "turbine blade", "score": 1.0}]
    assert found.offsets[1]["offsets"] == [{"term": "turbine blade", "offset": (0, 14)}]
    assert found.offsets[0]["offsets"] == [{"term": "turbine", "offset": (29, 36)}]
    # A sentence is never its own demonstration, and the most similar is the one
    # chosen: "Wind turbine blades." for "Turbine blades crack."
    prompts = dict(zip(texts[1:], client.prompts[2:]))
    assert (
        "Input: Wind turbine blades.\nOutput: Wind turbine blades"
        in (prompts["Turbine blades crack."])
    )


def test_knn_order():
    client = FakeClient()
    model = extractor(client, k=2)
    model.fit_strings(
        ["Heart failure. Wind turbine blade. Wind turbine."], {"heart failure": 1}
    )
    [prompt] = model.prompts([string_sentences("A wind turbine blade.")[0]])
    # The most similar is last, next to the sentence
    assert prompt.index("Input: Wind turbine.") < prompt.index(
        "Input: Wind turbine blade."
    )
    assert "Heart failure" not in prompt


def test_random_selection():
    train = ["One heart. Two hearts. Three hearts. Four hearts."]

    def prompts(seed):
        model = LLMTermExtractor(
            client=FakeClient(), selection="random", k=2, random_state=seed
        )
        model.fit_strings(train, {"heart": 1})
        return model.prompts(string_sentences("Five hearts. Six hearts."))

    assert prompts(0) == prompts(0)
    assert all(p.count("Input:") == 3 for p in prompts(0))
    model = LLMTermExtractor(client=FakeClient(), selection="random", k=10)
    model.fit_strings(train, {"heart": 1})
    # There are only four demonstrations, and none is the sentence itself
    [prompt] = model.prompts(string_sentences("Two hearts."))
    assert prompt.count("Input:") == 4


def test_zero_shot():
    client = FakeClient({"Heart failure.": "heart failure"})
    # Without demonstrations, no encoder is needed
    model = LLMTermExtractor(client=client, k=0)
    model.fit_strings(["Wind."], {"wind": 1})
    found = model.predict_strings(["Heart failure."])
    assert found.terms[0]["terms"] == [{"term": "heart failure", "score": 1.0}]
    assert client.prompts == [
        (
            "Find the domain-specific terms in the following text.\n\n"
            "Input: Heart failure.\nOutput:"
        )
    ]


def test_long_terms_and_n_jobs():
    text = "One two three four five."
    client = FakeClient({text: "one two three four five, two"})
    for n_jobs in (None, 2):
        model = LLMTermExtractor(client=client, k=0, n=4, n_jobs=n_jobs)
        model.fit_strings([], {})
        found = model.predict_strings([text, text])
        assert [d["terms"] for d in found.terms] == [
            [{"term": "two", "score": 1.0}]
        ] * 2


def test_custom_template():
    client = FakeClient()
    model = LLMTermExtractor(client=client, k=0, template=("List terms.", "S:", "T:"))
    model.fit_strings([], {})
    model.predict_strings(["Heart."])
    assert client.prompts == ["List terms.\n\nS: Heart.\nT:"]


def test_unfitted():
    with pytest.raises(RuntimeError, match="trained"):
        LLMTermExtractor(client=FakeClient()).predict_strings(["Heart."])


def test_invalid_parameters():
    with pytest.raises(ValueError, match="client"):
        LLMTermExtractor()
    with pytest.raises(ValueError, match="template"):
        LLMTermExtractor(client=FakeClient(), template=9)
    with pytest.raises(ValueError, match="template"):
        LLMTermExtractor(client=FakeClient(), template=("a", "b"))
    with pytest.raises(ValueError, match="k"):
        LLMTermExtractor(client=FakeClient(), k=-1)
    with pytest.raises(ValueError, match="selection"):
        LLMTermExtractor(client=FakeClient(), selection="bm25")


def test_sentence_term():
    sentence = Sentence("Heart  Failures", [(0, 5), (7, 15)], ["heart", "failure"])
    assert sentence.term(0, 2) == "heart failure"
    sentence.lemmas = None
    assert sentence.term(0, 2) == "heart failures"


@pytest.mark.skipif(not HAS_OPENAI, reason="requires the 'llm' extra")
def test_openai_client(tmp_path, monkeypatch):
    from termxtract.llm import OpenAIClient

    client = OpenAIClient(
        "model", base_url="http://localhost:1/v1", cache_dir=tmp_path, seed=1
    )
    requests = []

    def create(**request):
        requests.append(request)
        message = SimpleNamespace(content="<think>Hmm.</think>\nheart failure")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(client.client.chat.completions, "create", create)
    assert client.complete("Terms?") == "heart failure"
    assert client.complete("Terms?") == "heart failure"
    assert requests == [
        {
            "model": "model",
            "messages": [{"role": "user", "content": "Terms?"}],
            "temperature": 0,
            "max_tokens": 512,
            "seed": 1,
        }
    ]
    assert len(list(tmp_path.glob("*/*.json"))) == 1
