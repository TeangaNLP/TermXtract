import gzip

import numpy as np

from termxtract.embeddings import Encoder
from termxtract.wikipedia import WikipediaAPI, WikipediaDump, collect, lexicon

# A small Wikipedia: pages by id, as (namespace, title)
PAGES = {
    1: (0, "Heart failure"),
    2: (14, "Heart diseases"),
    3: (14, "Hidden category"),
    4: (0, "Cardiomyopathy"),
    5: (0, "Myocarditis"),
    6: (14, "Organ failure"),
    7: (0, "Kidney failure"),
    8: (1, "Heart failure"),
    9: (0, "Edema"),
}
HIDDEN = {3}
# Link targets by id, as (namespace, title)
TARGETS = {
    101: (14, "Heart diseases"),
    102: (14, "Hidden category"),
    103: (14, "Organ failure"),
    201: (0, "Dilated cardiomyopathy"),
    202: (0, "Heart"),
    203: (0, "Kidney"),
    204: (0, "Virus"),
    205: (4, "Wikipedia:Style"),
    206: (0, "Starling's law"),
}
# Category links as (page, sort key, target)
CATEGORIES = [
    (1, "HEART FAILURE", 101),
    (1, "HEART FAILURE", 102),
    (1, "HEART FAILURE", 103),
    (4, "CARDIOMYOPATHY", 101),
    (5, "MYOCARDITIS", 101),
    (7, "KIDNEY FAILURE", 103),
    (8, "HEART FAILURE", 101),
    (9, "EDEMA", 102),
]
# Links as (page, target)
LINKS = [(1, 202), (4, 201), (4, 202), (4, 205), (5, 204), (5, 206), (7, 203)]


def sql(value):
    """A value as written in a MySQL dump."""
    if isinstance(value, int | float):
        return str(value)
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def write_dump(directory):
    """Write the dumps, with each table in two statements."""

    def write(table, rows):
        statements = [rows[: len(rows) // 2], rows[len(rows) // 2 :]]
        with gzip.open(directory / f"enwiki-latest-{table}.sql.gz", "wt") as f:
            f.write(f"CREATE TABLE `{table}` (...);\n")
            for statement in statements:
                values = ",".join(
                    "(" + ",".join(sql(v) for v in row) + ")" for row in statement
                )
                f.write(f"INSERT INTO `{table}` VALUES {values};\n")

    def name(title):
        return title.replace(" ", "_")

    write(
        "page",
        [
            (
                i,
                ns,
                name(t),
                0,
                0,
                0.5,
                "20260101000000",
                "20260101000000",
                1,
                10,
                "wikitext",
            )
            for i, (ns, t) in sorted(PAGES.items())
        ],
    )
    write("page_props", [(i, "hiddencat", "", 0) for i in sorted(HIDDEN)])
    write(
        "categorylinks",
        [
            (page, key, "2026-01-01 00:00:00", "", "page", 1, target)
            for page, key, target in sorted(CATEGORIES)
        ],
    )
    write(
        "linktarget",
        [(i, ns, name(t)) for i, (ns, t) in sorted(TARGETS.items())],
    )
    write("pagelinks", [(page, 0, target) for page, target in sorted(LINKS)])


class FakeAPI(WikipediaAPI):
    """Answers queries from the small Wikipedia, with one link per response."""

    def __init__(self):
        super().__init__()
        self.requests = 0

    def get(self, params):
        self.requests += 1
        ids = {t: i for i, (ns, t) in PAGES.items() if ns == 0}
        if params.get("meta") == "siteinfo":
            return {"query": {"namespaces": {"14": {"name": "Category"}}}}
        if params.get("list") == "categorymembers":
            category = params["cmtitle"].split(":", 1)[1]
            target = next(i for i, (_, t) in TARGETS.items() if t == category)
            members = sorted(
                (key, page)
                for page, key, t in CATEGORIES
                if t == target and PAGES[page][0] == 0
            )
            return {
                "query": {
                    "categorymembers": [{"title": PAGES[p][1]} for _, p in members]
                }
            }
        if params.get("prop") == "categories":
            page = ids[params["titles"]]
            found = [
                {"title": "Category:" + TARGETS[t][1]}
                for p, _, t in CATEGORIES
                if p == page and not any(PAGES[h][1] == TARGETS[t][1] for h in HIDDEN)
            ]
            return {
                "query": {"pages": [{"title": params["titles"], "categories": found}]}
            }
        # Links, one per response, to follow continuations
        links = [
            (title, TARGETS[t][1])
            for title in params["titles"].split("|")
            for p, t in LINKS
            if p == ids[title] and TARGETS[t][0] == 0
        ]
        start = int(params.get("plcontinue", 0))
        response = {
            "query": {"pages": [{"title": t} for t in params["titles"].split("|")]}
        }
        if start < len(links):
            title, link = links[start]
            response["query"]["pages"] = [{"title": title, "links": [{"title": link}]}]
        if start + 1 < len(links):
            response["continue"] = {"plcontinue": str(start + 1), "continue": "||"}
        return response


EXPECTED = ["dilated cardiomyopathy", "heart", "starling's law", "virus", "kidney"]


def test_collect():
    members = {"A": ["a1", "a2"], "B": ["b1"]}
    links = {"a1": ["X", "Y"], "a2": ["Y", "Z"], "b1": ["W"]}
    fetched = []

    def get_links(pages):
        for page in pages:
            fetched.append(page)
            yield links[page]

    assert collect(["A", "B"], members.get, get_links, 10) == ["x", "y", "z", "w"]
    fetched.clear()
    # Stops after the article that reaches the limit, without fetching more
    assert collect(["A", "B"], members.get, get_links, 2) == ["x", "y"]
    assert fetched == ["a1"]


def test_dump(tmp_path):
    write_dump(tmp_path)
    dump = WikipediaDump(tmp_path)
    found = dump.candidates(["Heart failure", "Missing"])
    # Heart diseases, by sort key, then Organ failure; Edema is in a hidden category,
    # Heart failure's talk page is not an article and Wikipedia:Style not an article
    assert found == {"Heart failure": EXPECTED, "Missing": []}
    # The candidates are cached
    (tmp_path / "enwiki-latest-pagelinks.sql.gz").unlink()
    assert dump.candidates(["Heart failure"]) == {"Heart failure": EXPECTED}


def test_api():
    api = FakeAPI()
    assert api.candidates(["Heart failure"]) == {"Heart failure": EXPECTED}


def test_api_cache(tmp_path, monkeypatch):
    api = WikipediaAPI(cache_dir=tmp_path)
    responses = iter([{"query": {"pages": []}}])
    sent = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return b'{"query": {"pages": []}}'

    def urlopen(request, timeout):
        sent.append(request.full_url)
        next(responses)
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert api.get({"titles": "Heart failure"}) == {"query": {"pages": []}}
    assert api.get({"titles": "Heart failure"}) == {"query": {"pages": []}}
    assert len(sent) == 1
    assert "titles=Heart+failure" in sent[0]


class FakeEncoder(Encoder):
    """Embeds a text by the words it shares with a fixed vocabulary."""

    VOCABULARY = ("heart", "failure", "kidney", "cardiomyopathy", "dilated")

    def __init__(self):
        pass

    def encode(self, texts, masks=None):
        vectors = np.array(
            [[0.1] + [float(w in text) for w in self.VOCABULARY] for text in texts]
        )
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_lexicon():
    found = lexicon(
        FakeAPI(),
        "Heart failure",
        ["Cardiomyopathy", "heart", "kidney failure"],
        encoder=FakeEncoder(),
        threshold=0.6,
    )
    # A term is not its own replacement, and terms without replacements are left out
    assert found == {
        "cardiomyopathy": ["dilated cardiomyopathy"],
        "kidney failure": ["kidney"],
    }
