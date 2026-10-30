import importlib.util
import inspect

import pytest

from termxtract import SupervisedTermExtractor, TermExtractor

HAS_TRANSFORMERS = all(
    importlib.util.find_spec(module) is not None
    for module in ("torch", "transformers", "sacremoses")
)


def wrapper_params(wrapper):
    return set(inspect.signature(wrapper.__init__).parameters) - {"self", "method"}


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
