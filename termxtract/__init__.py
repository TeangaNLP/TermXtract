"""TermXtract: automatic term extraction with unsupervised and supervised methods.

The public classes are loaded lazily, so ``import termxtract`` does not pull in
heavy dependencies (e.g. torch/transformers for the supervised extractors) until
they are actually used.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .supervised.term_extractor import SupervisedTermExtractor
    from .unsupervised.term_extractor import TermExtractor
    from .utils import ATEResults

try:
    __version__ = version("termxtract")
except PackageNotFoundError:  # Running from a source tree that isn't installed
    __version__ = "0+unknown"

_LAZY_IMPORTS = {
    "TermExtractor": ".unsupervised.term_extractor",
    "SupervisedTermExtractor": ".supervised.term_extractor",
    "ATEResults": ".utils",
}

__all__ = ["ATEResults", "SupervisedTermExtractor", "TermExtractor"]


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        try:
            module = import_module(_LAZY_IMPORTS[name], __name__)
        except AttributeError as e:
            # Otherwise `from termxtract import X` reports this as "cannot import name 'X'"
            # and hides the underlying error.
            raise ImportError(
                f"Error while importing {name!r} from {__name__!r}"
            ) from e
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
