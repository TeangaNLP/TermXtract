"""
Domain lexicons from Wikipedia for Wiki-TermDA (Banerjee, 2025, Algorithm 1): the links
of the articles in the categories of a domain's article, mapped to the gold terms they
are similar to.

The articles and links come from the SQL dumps of a Wikipedia (`WikipediaDump`), which
are a fixed snapshot, or from its API (`WikipediaAPI`), which needs no download. Both
take categories, articles and links in the same order, so that given the same Wikipedia
they find the same candidates.
"""

import gzip
import hashlib
import json
import re
import shutil
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Protocol

import numpy as np

from . import __version__
from .datasets.base import data_home
from .embeddings import Encoder

USER_AGENT = f"termxtract/{__version__} (https://github.com/TeangaNLP/TermXtract)"
"""The User-Agent of requests to Wikimedia, which asks that clients name themselves."""

TABLES = ("page", "page_props", "categorylinks", "linktarget", "pagelinks")
"""The SQL dumps read by `WikipediaDump`."""

QUALIFIER = re.compile(r"\s+\([^()]*\)$")
"""The qualifier of an article title, as in "Mercury (planet)"."""


class WikipediaSource(Protocol):
    """A Wikipedia to collect candidate terms from, such as `WikipediaDump`."""

    def candidates(
        self, titles: list[str], max_candidates: int = 2000
    ) -> dict[str, list[str]]:
        """
        Collect the candidate terms of the domains of some articles.

        Args:
            titles (List[str]): The titles of the articles, such as "Heart failure".
            max_candidates (int): The number of candidates after which no more articles
                are read.

        Returns:
            Dict[str, List[str]]: The candidates of each title, in lower case.
        """
        ...


def collect(
    categories: list[str],
    members: Callable[[str], list[str]],
    links: Callable[[list[str]], Iterator[list[str]]],
    max_candidates: int,
) -> list[str]:
    """
    Collect candidates as in Algorithm 1 of Banerjee (2025): the links, in lower case,
    of the articles in each category in turn, until there are `max_candidates`.

    Args:
        categories (List[str]): The categories of the domain's article, in order.
        members (Callable[[str], List[str]]): The articles of a category, in order.
        links (Callable[[List[str]], Iterator[List[str]]]): The links of each of some
            articles, in order, generated lazily so that no more are fetched than are
            read.
        max_candidates (int): The number of candidates after which no more articles
            are read.

    Returns:
        List[str]: The candidates, without duplicates, in the order found.
    """
    found: dict[str, None] = {}
    for category in categories:
        if len(found) >= max_candidates:
            break
        for article_links in links(members(category)):
            found.update(dict.fromkeys(link.lower() for link in article_links))
            if len(found) >= max_candidates:
                break
    return list(found)


def lexicon(
    source: WikipediaSource,
    title: str,
    terms: Iterable[str],
    encoder: Encoder | None = None,
    threshold: float = 0.75,
    max_candidates: int = 2000,
) -> dict[str, list[str]]:
    """
    Build the lexicon of a domain for Wiki-TermDA (Banerjee, 2025, Algorithm 1).

    The candidates are the links of the articles in the categories of the domain's
    article. Each term is mapped to the candidates whose embedding has a cosine
    similarity above `threshold` with its own. Qualifiers in brackets are dropped from
    the candidates, so that "Mercury (planet)" is "mercury", and a term is not its own
    replacement.

    Args:
        source (WikipediaSource): The Wikipedia, a `WikipediaDump` or a `WikipediaAPI`.
        title (str): The title of the domain's article, such as "Heart failure".
        terms (Iterable[str]): The gold terms.
        encoder (Optional[Encoder]): The encoder of the terms, or None for `Encoder()`,
            all-MiniLM-L6-v2 as in Banerjee (2025), which needs the 'transformers'
            extra.
        threshold (float): The minimum similarity of a replacement.
        max_candidates (int): The number of candidates after which no more articles
            are read.

    Returns:
        Dict[str, List[str]]: The replacements of each term, in lower case; terms
        without replacements are left out.
    """
    terms = list(dict.fromkeys(term.lower() for term in terms))
    found = source.candidates([title], max_candidates)[title]
    candidates = list(dict.fromkeys(QUALIFIER.sub("", c) for c in found))
    if not terms or not candidates:
        return {}
    if encoder is None:
        encoder = Encoder()
    similarity = (
        encoder.encode([term.split() for term in terms])
        @ encoder.encode([c.split() for c in candidates]).T
    )
    mapping = {}
    for term, row in zip(terms, similarity):
        similar = [c for c, s in zip(candidates, row) if s > threshold and c != term]
        if similar:
            mapping[term] = similar
    return mapping


class WikipediaAPI:
    """
    A Wikipedia read through its API. A domain takes a few dozen requests, sent one at
    a time; they can be cached on disk, so that they are only sent once.
    """

    def __init__(
        self,
        language: str = "en",
        cache_dir: str | Path | None = None,
        user_agent: str = USER_AGENT,
    ):
        """
        Args:
            language (str): The language code of the Wikipedia, such as "en".
            cache_dir (Optional[Union[str, Path]]): A directory to keep each response
                in, or None to cache nothing.
            user_agent (str): The User-Agent of the requests.
        """
        self.url = f"https://{language}.wikipedia.org/w/api.php"
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.user_agent = user_agent
        self.category_namespace: str | None = None

    def query(self, **params: str) -> Iterator[dict]:
        """
        Send a query, following its continuations.

        Args:
            **params: The parameters of the query.

        Returns:
            Iterator[dict]: The `query` part of each response.
        """
        params = {"action": "query", "format": "json", "formatversion": "2", **params}
        cont: dict[str, str] = {}
        while True:
            response = self.get({**params, **cont})
            yield response.get("query", {})
            if "continue" not in response:
                return
            cont = response["continue"]

    def get(self, params: dict[str, str]) -> dict:
        """Send a request, from the cache if it has been sent before."""
        url = f"{self.url}?{urllib.parse.urlencode(sorted(params.items()))}"
        path = None
        if self.cache_dir is not None:
            name = hashlib.sha256(url.encode()).hexdigest()
            path = self.cache_dir / name[:2] / f"{name}.json"
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.load(response)
        if "error" in data:
            raise RuntimeError(f"Wikipedia API error: {data['error']}")
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(".tmp")
            partial.write_text(json.dumps(data), encoding="utf-8")
            partial.replace(path)
        return data

    def categories(self, title: str) -> list[str]:
        """The categories of an article that are not hidden, by title, without the
        namespace."""
        found = []
        for query in self.query(
            prop="categories", titles=title, clshow="!hidden", cllimit="max"
        ):
            for page in query.get("pages", []):
                found += [
                    c["title"].split(":", 1)[1] for c in page.get("categories", [])
                ]
        return sorted(found, key=lambda t: t.replace(" ", "_").encode())

    def members(self, category: str) -> list[str]:
        """The articles of a category, by sort key."""
        if self.category_namespace is None:
            info = next(self.query(meta="siteinfo", siprop="namespaces"))
            self.category_namespace = info["namespaces"]["14"]["name"]
        return [
            member["title"]
            for query in self.query(
                list="categorymembers",
                cmtitle=f"{self.category_namespace}:{category}",
                cmnamespace="0",
                cmtype="page",
                cmlimit="max",
            )
            for member in query.get("categorymembers", [])
        ]

    def links(self, titles: list[str], batch: int = 10) -> Iterator[list[str]]:
        """The links to articles of each article, by title, fetched `batch` articles
        at a time."""
        for start in range(0, len(titles), batch):
            chunk = titles[start : start + batch]
            found: dict[str, list[str]] = {title: [] for title in chunk}
            for query in self.query(
                prop="links", titles="|".join(chunk), plnamespace="0", pllimit="max"
            ):
                for page in query.get("pages", []):
                    found.setdefault(page["title"], []).extend(
                        link["title"] for link in page.get("links", [])
                    )
            for title in chunk:
                yield sorted(found[title], key=lambda t: t.replace(" ", "_").encode())

    def candidates(
        self, titles: list[str], max_candidates: int = 2000
    ) -> dict[str, list[str]]:
        """
        Collect the candidate terms of the domains of some articles.

        Args:
            titles (List[str]): The titles of the articles, such as "Heart failure".
            max_candidates (int): The number of candidates after which no more articles
                are read.

        Returns:
            Dict[str, List[str]]: The candidates of each title, in lower case.
        """
        return {
            title: collect(
                self.categories(title), self.members, self.links, max_candidates
            )
            for title in titles
        }


ESCAPES = {b"0": b"\0", b"n": b"\n", b"r": b"\r", b"Z": b"\x1a", b"t": b"\t"}
"""The escapes of MySQL strings other than an escaped character standing for itself."""

STRING = rb"'((?:[^'\\]|\\.)*)'"
"""A MySQL string in a dump."""

ROWS = {
    "page": re.compile(rb"\((\d+),(-?\d+)," + STRING + rb","),
    "page_props": re.compile(rb"\((\d+),'(hiddencat)',"),
    "categorylinks": re.compile(
        rb"\((\d+)," + STRING + rb",'[^']*'," + STRING + rb",'(\w+)',\d+,(\d+)\)"
    ),
    "linktarget": re.compile(rb"\((\d+),(-?\d+)," + STRING + rb"\)"),
}
"""The start of a row of each table but `pagelinks`, with the columns read."""

BLOCK = 1 << 26
"""The size of the blocks in which the dumps are read."""

DIGITS = bytes(c if chr(c).isdigit() and c < 128 else 32 for c in range(256))
"""A table that turns every byte but the digits into a space."""


def unescape(value: bytes) -> bytes:
    """Undo the escapes of a MySQL string."""
    return re.sub(rb"\\(.)", lambda m: ESCAPES.get(m.group(1), m.group(1)), value)


def title(value: bytes) -> str:
    """The title of a page in a dump, with spaces."""
    return unescape(value).decode("utf-8", "replace").replace("_", " ")


class WikipediaDump:
    """
    A Wikipedia read from its SQL dumps (https://dumps.wikimedia.org): the `page`,
    `page_props`, `categorylinks`, `linktarget` and `pagelinks` tables, about 14 GB
    for English. Each table is read a few times for each call to `candidates`, so the
    candidates of every domain should be collected in one call; they are cached in the
    directory of the dumps.
    """

    def __init__(
        self,
        directory: str | Path | None = None,
        language: str = "en",
        date: str = "latest",
    ):
        """
        Args:
            directory (Optional[Union[str, Path]]): The directory of the dumps, or None
                for `wikipedia` in the data directory of `termxtract.datasets`.
            language (str): The language code of the Wikipedia, such as "en".
            date (str): The date of the dumps, such as "20261101", or "latest".
        """
        self.directory = (
            data_home() / "wikipedia" if directory is None else Path(directory)
        )
        self.wiki = f"{language}wiki"
        self.date = date

    def path(self, table: str) -> Path:
        """The file of the dump of a table."""
        return self.directory / f"{self.wiki}-{self.date}-{table}.sql.gz"

    def url(self, table: str) -> str:
        """The URL of the dump of a table."""
        return (
            f"https://dumps.wikimedia.org/{self.wiki}/{self.date}/"
            f"{self.path(table).name}"
        )

    def download(self, user_agent: str = USER_AGENT) -> None:
        """
        Download the dumps that are missing.

        Args:
            user_agent (str): The User-Agent of the requests.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        for table in TABLES:
            path = self.path(table)
            if path.exists():
                continue
            partial = path.with_suffix(".partial")
            request = urllib.request.Request(
                self.url(table), headers={"User-Agent": user_agent}
            )
            with urllib.request.urlopen(request) as response, partial.open("wb") as f:
                shutil.copyfileobj(response, f, 1 << 20)
            partial.replace(path)

    def blocks(self, table: str) -> Iterator[bytes]:
        """
        Read the dump of a table in blocks of whole lines.

        Args:
            table (str): The table.

        Returns:
            Iterator[bytes]: Blocks of about 64 MB.

        Raises:
            FileNotFoundError: If the dump has not been downloaded.
        """
        path = self.path(table)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing: download it from {self.url(table)}, or call "
                "download()."
            )
        rest = b""
        with gzip.open(path, "rb") as f:
            while block := f.read(BLOCK):
                block = rest + block
                cut = block.rfind(b"\n") + 1
                rest = block[cut:]
                yield block[:cut]
        if rest:
            yield rest

    def rows(
        self,
        table: str,
        keys: set[int] | None = None,
        needles: Iterable[bytes] = (),
    ) -> Iterator[tuple[bytes, ...]]:
        """
        Read the rows of a table, one per line as the dumps write them, whose first
        column is one of `keys` or that contain one of `needles`.

        Args:
            table (str): The table.
            keys (Optional[Set[int]]): The first columns of the rows wanted, or None.
            needles (Iterable[bytes]): Text of the rows wanted, which should be rare: a
                row that has both a key and a needle is read twice.

        Returns:
            Iterator[Tuple[bytes, ...]]: The columns of `ROWS` of each row read.
        """
        pattern = ROWS[table]
        wanted = {b"%d" % key for key in keys or ()}
        needles = list(needles)
        for block in self.blocks(table):
            for needle in needles:
                found = block.find(needle)
                while found >= 0:
                    start = block.rfind(b"\n", 0, found) + 1
                    end = block.find(b"\n", found)
                    if match := pattern.match(block, start, end):
                        yield match.groups()
                    found = block.find(needle, end)
            if wanted:
                for line in block.split(b"\n"):
                    if line[1 : line.find(b",")] in wanted and (
                        match := pattern.match(line)
                    ):
                        yield match.groups()

    def links(self, pages: set[int]) -> Iterator[tuple[int, int]]:
        """
        Read the links from some pages to articles, parsing the numbers of `pagelinks`
        with numpy, as it has billions of rows.

        Args:
            pages (Set[int]): The ids of the pages.

        Returns:
            Iterator[Tuple[int, int]]: The id of each page and of the link target of
            each of its links.
        """
        wanted = np.array(sorted(pages), dtype=np.int64)
        started = False
        for block in self.blocks("pagelinks"):
            # Only the rows have numbers: skip the header before them and the
            # trailer after them, which have version numbers and dates. A block
            # starts a line, so each line starts after a newline
            text = b"\n" + block
            start = 0
            if not started:
                start = text.find(b"\n(")
                if start < 0:
                    continue
                started = True
            end = text.find(b"\n/*", start)
            numbers = np.fromstring(  # type: ignore[call-overload]
                text[start : len(text) if end < 0 else end].translate(DIGITS),
                dtype=np.int64,
                sep=" ",
            ).reshape(-1, 3)
            rows = numbers[np.isin(numbers[:, 0], wanted) & (numbers[:, 1] == 0)]
            yield from zip(rows[:, 0].tolist(), rows[:, 2].tolist())
            if end >= 0:
                return

    def candidates(
        self, titles: list[str], max_candidates: int = 2000
    ) -> dict[str, list[str]]:
        """
        Collect the candidate terms of the domains of some articles.

        Args:
            titles (List[str]): The titles of the articles, such as "Heart failure".
            max_candidates (int): The number of candidates after which no more articles
                are read.

        Returns:
            Dict[str, List[str]]: The candidates of each title, in lower case.
        """
        cache = self.directory / f"{self.wiki}-{self.date}-candidates.json"
        cached = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
        key = str(max_candidates)
        missing = [t for t in titles if t not in cached.get(key, {})]
        if missing:
            cached.setdefault(key, {}).update(self._candidates(missing, max_candidates))
            cache.write_text(json.dumps(cached), encoding="utf-8")
        return {t: cached[key][t] for t in titles}

    def _candidates(
        self, titles: list[str], max_candidates: int
    ) -> dict[str, list[str]]:
        """Collect candidates from the dumps."""
        article_ids = {
            title(name): int(page)
            for page, namespace, name in self.rows("page", needles=_needles(titles, 0))
            if namespace == b"0" and title(name) in titles
        }

        # The categories of the articles that are not hidden
        targets: dict[int, list[int]] = {}
        for page, _, _, _, target in self.rows(
            "categorylinks", set(article_ids.values())
        ):
            targets.setdefault(int(page), []).append(int(target))
        names_of = self._targets({t for ts in targets.values() for t in ts}, 14)
        hidden = {int(page) for page, _ in self.rows("page_props", needles=[b"hidden"])}
        hidden_titles = {
            title(name)
            for page, namespace, name in self.rows(
                "page", needles=_needles(names_of.values(), 14)
            )
            if namespace == b"14" and int(page) in hidden
        }
        categories = {
            t: sorted(
                (
                    names_of[target]
                    for target in targets.get(page, [])
                    if target in names_of and names_of[target] not in hidden_titles
                ),
                key=lambda c: c.replace(" ", "_").encode(),
            )
            for t, page in article_ids.items()
        }
        category_ids = {name: target for target, name in names_of.items()}

        # The pages of the categories, by sort key
        wanted = {category_ids[c] for cs in categories.values() for c in cs}
        listed: dict[int, list[tuple[bytes, int]]] = {}
        for page, sortkey, _, kind, target in self.rows(
            "categorylinks", needles=(b",%d)" % t for t in wanted)
        ):
            if kind == b"page" and int(target) in wanted:
                listed.setdefault(int(target), []).append(
                    (unescape(sortkey), int(page))
                )
        member_ids = {page for pages in listed.values() for _, page in pages}
        articles = {
            int(page): title(name)
            for page, namespace, name in self.rows("page", member_ids)
            if namespace == b"0" and int(page) in member_ids
        }
        members = {
            category: [
                articles[page]
                for _, page in sorted(listed.get(category_ids[category], []))
                if page in articles
            ]
            for cs in categories.values()
            for category in cs
        }

        # The links of the pages
        link_targets: dict[int, list[int]] = {}
        for source, link in self.links(set(articles)):
            link_targets.setdefault(source, []).append(link)
        names_of = self._targets({t for ts in link_targets.values() for t in ts}, 0)
        ids = {name: page for page, name in articles.items()}
        links = {
            name: sorted(
                (names_of[t] for t in link_targets.get(page, []) if t in names_of),
                key=lambda link: link.replace(" ", "_").encode(),
            )
            for name, page in ids.items()
        }
        return {
            t: collect(
                categories[t],
                lambda c: members[c],
                lambda pages: (links[p] for p in pages),
                max_candidates,
            )
            for t in titles
            if t in categories
        } | {t: [] for t in titles if t not in categories}

    def _targets(self, ids: set[int], namespace: int) -> dict[int, str]:
        """The titles of link targets in a namespace."""
        return {
            int(target): title(name)
            for target, ns, name in self.rows("linktarget", ids)
            if int(ns) == namespace and int(target) in ids
        }


def _needles(titles: Iterable[str], namespace: int) -> list[bytes]:
    """The text of the rows of the `page` table for some titles in a namespace."""
    return [
        b",%d,'" % namespace
        + t.replace(" ", "_").encode().replace(b"\\", b"\\\\").replace(b"'", b"\\'")
        + b"',"
        for t in titles
    ]
