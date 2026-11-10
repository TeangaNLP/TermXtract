import importlib.util
import inspect
import re

import pytest
from teanga import Corpus

from termxtract import SupervisedTermExtractor, TermExtractor

HAS_TRANSFORMERS = all(
    importlib.util.find_spec(module) is not None
    for module in ("torch", "transformers", "sacremoses")
)


def wrapper_params(wrapper):
    return set(inspect.signature(wrapper.__init__).parameters) - {
        "self",
        "method",
        *getattr(wrapper, "FILTER_PARAMS", ()),
    }


@pytest.mark.parametrize("method", list(TermExtractor.METHODS))
def test_unsupervised_registry_params(method):
    extractor_class, param_names = TermExtractor.METHODS[method]
    assert set(param_names) <= wrapper_params(TermExtractor)
    assert set(param_names) <= set(inspect.signature(extractor_class).parameters)


@pytest.mark.parametrize("method", list(SupervisedTermExtractor.METHODS))
def test_supervised_registry_params(method):
    if method == "token-classification" and not HAS_TRANSFORMERS:
        pytest.skip("requires the optional 'transformers' extra")
    load_extractor_class, param_names = SupervisedTermExtractor.METHODS[method]
    assert set(param_names) <= wrapper_params(SupervisedTermExtractor)
    assert set(param_names) <= set(inspect.signature(load_extractor_class()).parameters)


def test_every_wrapper_param_is_used():
    used = {name for _, names in TermExtractor.METHODS.values() for name in names}
    assert used == wrapper_params(TermExtractor)
    used = {
        name for _, names in SupervisedTermExtractor.METHODS.values() for name in names
    }
    assert used == wrapper_params(SupervisedTermExtractor)


@pytest.mark.parametrize("wrapper", [TermExtractor, SupervisedTermExtractor])
def test_unknown_method_lists_available_methods(wrapper):
    with pytest.raises(ValueError, match="Available methods: .*") as error:
        wrapper(method="no-such-method")
    for method in wrapper.METHODS:
        assert method in str(error.value)


def test_parameters_reach_the_extractor():
    extractor = TermExtractor(method="combobasic", alpha=0, beta=2, n=3)
    assert extractor.extractor.alpha == 0
    assert extractor.extractor.beta == 2
    assert extractor.extractor.n == 3


def test_rake_requires_stoplist_and_delimiters():
    with pytest.raises(ValueError, match="RAKE requires"):
        TermExtractor(method="rake", stoplist=["the"])


def corpus_with_layer(layer):
    texts = [
        "Machine learning is a field of artificial intelligence.",
        "Deep learning is a subfield of machine learning.",
        "Natural language processing applies machine learning to text.",
    ]
    corpus = Corpus()
    corpus.add_layer_meta("text")
    corpus.add_layer_meta(layer, layer_type="span", base="text")
    for text in texts:
        doc = corpus.add_doc(text)
        doc[layer] = [match.span() for match in re.finditer(r"\w+", text)]
    return corpus


@pytest.mark.parametrize("method", [m for m in TermExtractor.METHODS if m != "rake"])
def test_unsupervised_words_layer(method):
    def extract(layer, **params):
        corpus = corpus_with_layer(layer)
        if method in ("weirdness", "domainpertinence", "relevance"):
            params["reference_corpus"] = corpus_with_layer(layer)
        return TermExtractor(method=method, n=2, **params).extract(corpus)

    expected = extract("words")
    # "tokens" is used when there is no "words" layer, or when named
    assert extract("tokens").terms == expected.terms
    assert extract("segments", words_layer="segments").terms == expected.terms
    assert expected.terms[0]["terms"]


def test_supervised_words_layer():
    labels = {"machine learning": 1, "deep learning": 1, "is a": 0, "of machine": 0}

    def extract(layer):
        return SupervisedTermExtractor(
            method="roger", n=2, random_state=0, generations=5, runs=2
        ).extract(corpus_with_layer(layer), labels)

    assert extract("tokens").terms == extract("words").terms
