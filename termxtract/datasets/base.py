import hashlib
import inspect
import logging
import os
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from teanga import Corpus

logger = logging.getLogger(__name__)

GOLD_LAYER = "terms"
"""The span layer that holds the gold terms of a loaded dataset."""

CATEGORY_LAYER = "categories"
"""The sequence layer, based on `GOLD_LAYER`, that holds the class of each gold term,
for datasets that classify their terms."""

SOURCE_LAYER = "source"
"""The characters layer holding the name of the file or record each document comes
from, such as `htfl_en_001`."""

SENTENCES_LAYER = "sentences"
"""The span layer of the sentences of each document."""


@dataclass(frozen=True)
class Resource:
    """
    A file that a dataset downloads.

    Attributes:
        url (str): Where to download the file from, pinned to a fixed version.
        sha256 (str): The SHA-256 checksum of the file, in hexadecimal.
        filename (str): The name of the file in the dataset's cache directory.
    """

    url: str
    sha256: str
    filename: str


@dataclass(frozen=True)
class DatasetInfo:
    """
    A dataset that `load` can load.

    Attributes:
        name (str): The name passed to `load`.
        description (str): A one-line description.
        languages (Tuple[str, ...]): ISO 639 codes of the languages of the texts.
        domains (Tuple[str, ...]): The domains of the texts.
        license (str): The license of the dataset.
        citation (str): The publication to cite when using the dataset.
        homepage (str): Where the dataset is described.
        resources (Tuple[Resource, ...]): The files to download.
        loader (Callable[..., Corpus]): A function that takes the paths of the
            downloaded resources as positional arguments, in order, then a `split`
            keyword argument and the options of the dataset as keyword-only arguments,
            and returns a Teanga corpus with the gold terms in a `GOLD_LAYER` span
            layer.
        splits (Tuple[str, ...]): The values `split` can take, or `()` if the dataset
            has no splits.
    """

    name: str
    description: str
    languages: tuple[str, ...]
    domains: tuple[str, ...]
    license: str
    citation: str
    homepage: str
    resources: tuple[Resource, ...]
    loader: Callable[..., Corpus]
    splits: tuple[str, ...] = ()

    @property
    def options(self) -> tuple[str, ...]:
        """The options of the dataset: the keyword-only arguments of `loader`, besides
        `split`."""
        return tuple(
            p.name
            for p in inspect.signature(self.loader).parameters.values()
            if p.kind is inspect.Parameter.KEYWORD_ONLY and p.name != "split"
        )


_REGISTRY: dict[str, DatasetInfo] = {}


def register(info: DatasetInfo) -> DatasetInfo:
    """
    Make a dataset available to `load`.

    Args:
        info (DatasetInfo): The dataset.

    Returns:
        DatasetInfo: `info`, unchanged.

    Raises:
        ValueError: If a dataset with the same name is already registered.
    """
    if info.name in _REGISTRY:
        raise ValueError(f"A dataset named {info.name!r} is already registered.")
    _REGISTRY[info.name] = info
    return info


def list_datasets() -> list[DatasetInfo]:
    """
    List the datasets that `load` can load.

    Returns:
        List[DatasetInfo]: The datasets, sorted by name.
    """
    return [_REGISTRY[name] for name in sorted(_REGISTRY)]


def info(name: str) -> DatasetInfo:
    """
    Describe a dataset.

    Args:
        name (str): The name of the dataset.

    Returns:
        DatasetInfo: The dataset.

    Raises:
        ValueError: If there is no dataset with this name.
    """
    if name not in _REGISTRY:
        available = ", ".join(map(repr, sorted(_REGISTRY))) or "none"
        raise ValueError(f"Unknown dataset: {name!r}. Available datasets: {available}.")
    return _REGISTRY[name]


def data_home(path: str | os.PathLike | None = None) -> Path:
    """
    Get the directory where datasets are cached.

    Args:
        path (Optional[Union[str, PathLike]]): A directory to use instead of the
            default.

    Returns:
        Path: `path` if given, otherwise the `TERMXTRACT_DATA` environment variable if
        set, otherwise `termxtract` in the user's cache directory (`$XDG_CACHE_HOME`, or
        `~/.cache`).
    """
    if path is not None:
        return Path(path).expanduser()
    if "TERMXTRACT_DATA" in os.environ:
        return Path(os.environ["TERMXTRACT_DATA"]).expanduser()
    cache = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(cache).expanduser() / "termxtract"


def download(
    name: str, home: str | os.PathLike | None = None, force: bool = False
) -> list[Path]:
    """
    Download the files of a dataset, unless they are already cached.

    Each file is checked against its SHA-256 checksum when it is downloaded, and is
    stored as `<data home>/<dataset name>/<filename>`.

    Args:
        name (str): The name of the dataset.
        home (Optional[Union[str, PathLike]]): The cache directory, as for `data_home`.
        force (bool): Download the files again even if they are cached.

    Returns:
        List[Path]: The paths of the files, in the order of `DatasetInfo.resources`.

    Raises:
        ValueError: If there is no dataset with this name, or a downloaded file does not
            match its checksum. The file is then not kept.
    """
    dataset = info(name)
    directory = data_home(home) / dataset.name
    paths = []
    for resource in dataset.resources:
        path = directory / resource.filename
        if force or not path.exists():
            _fetch(resource, path)
        paths.append(path)
    return paths


def load(
    name: str,
    split: str | None = None,
    home: str | os.PathLike | None = None,
    download_if_missing: bool = True,
    **options: Any,
) -> Corpus:
    """
    Load a dataset as a Teanga corpus, downloading it on first use.

    The gold terms are in the `terms` span layer (`GOLD_LAYER`), so the corpus can be
    passed both to an extractor and, as the gold standard, to `termxtract.evaluate`.
    Datasets that classify their terms also have a `categories` layer
    (`CATEGORY_LAYER`).

    The datasets are not distributed with TermXtract. They are downloaded from their
    original sources, under their own licenses, which `info(name).license` gives.

    Args:
        name (str): The name of the dataset; see `list_datasets`.
        split (Optional[str]): The part of the dataset to load, one of
            `info(name).splits`, or None for all of it.
        home (Optional[Union[str, PathLike]]): The cache directory, as for `data_home`.
        download_if_missing (bool): Download the dataset if it is not cached. If False,
            a missing dataset raises `FileNotFoundError`.
        **options (Any): Options of the dataset, listed in `info(name).options`, such as
            `language` or `domain`.

    Returns:
        Corpus: The dataset.

    Raises:
        ValueError: If the dataset, split or an option is unknown.
        FileNotFoundError: If the dataset is not cached and `download_if_missing` is
            False.
    """
    dataset = info(name)
    if split is not None and split not in dataset.splits:
        splits = ", ".join(map(repr, dataset.splits)) or "none"
        raise ValueError(
            f"Unknown split for {name!r}: {split!r}. Available splits: {splits}."
        )
    unknown = sorted(set(options) - set(dataset.options))
    if unknown:
        known = ", ".join(dataset.options) or "none"
        raise ValueError(
            f"Unknown options for {name!r}: {', '.join(unknown)}. Available options: "
            f"{known}."
        )
    if download_if_missing:
        paths = download(name, home)
    else:
        directory = data_home(home) / dataset.name
        paths = [directory / resource.filename for resource in dataset.resources]
        missing = [path for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"The {name!r} dataset is not cached: {missing[0]} is missing. Call "
                f"termxtract.datasets.download({name!r}) or pass "
                "download_if_missing=True."
            )
    return dataset.loader(*paths, split=split, **options)


def gold_terms(
    corpus: Corpus, layer: str = GOLD_LAYER, by_document: bool = False
) -> set[str] | dict[str, set[str]]:
    """
    Get the gold terms of a loaded dataset.

    Args:
        corpus (Corpus): A corpus from `load`.
        layer (str): The span layer of gold terms.
        by_document (bool): Return the terms of each document instead of those of the
            whole corpus.

    Returns:
        Union[Set[str], Dict[str, Set[str]]]: The distinct gold terms, lowercased and
        with whitespace collapsed, or a dictionary from document IDs to them.
    """
    by_doc = {
        doc_id: {
            " ".join(term.lower().split())
            for term in corpus.doc_by_id(doc_id)[layer].text
        }
        for doc_id in corpus.doc_ids
    }
    if by_document:
        return by_doc
    return set().union(*by_doc.values())


def _fetch(resource: Resource, path: Path) -> None:
    """Download a resource to `path`, checking its checksum."""
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s to %s", resource.url, path)
    digest = hashlib.sha256()
    # Write to a temporary file in the same directory, so a failed or interrupted
    # download never leaves a partial file under the final name.
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with (
            os.fdopen(fd, "wb") as out,
            urllib.request.urlopen(resource.url) as response,
        ):
            while chunk := response.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != resource.sha256.lower():
            raise ValueError(
                f"The file downloaded from {resource.url} does not match its checksum: "
                f"expected SHA-256 {resource.sha256}, got {digest.hexdigest()}. The "
                "file may have changed at its source."
            )
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
