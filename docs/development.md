# Development

To work on TermXtract, clone it and install it with [Poetry](https://python-poetry.org/), with the development tools and the optional transformer dependencies:

```bash
git clone https://github.com/TeangaNLP/TermXtract.git
cd TermXtract
poetry install --with dev --extras transformers

poetry run pytest                              # run the test suite
poetry run pytest --cov                        # ...with a coverage report
poetry run ruff check termxtract tests         # lint
poetry run ruff format termxtract tests        # format
poetry run mypy termxtract                     # type check
poetry run mkdocs serve                        # preview the docs locally
```

CI runs all of these on every pull request, with the tests on Python 3.10–3.12, on Linux, macOS and Windows, and against the oldest dependency versions `pyproject.toml` allows. The codebase was reformatted in one commit, which `git blame` can skip with `git config blame.ignoreRevsFile .git-blame-ignore-revs`.

The scripts of `benchmarks/` reproduce the results of [BENCHMARKS.md](https://github.com/TeangaNLP/TermXtract/blob/main/BENCHMARKS.md); see each script for its options and requirements.

## Releasing

Version numbers follow [Semantic Versioning](https://semver.org/): bump the patch version for bug fixes, the minor version for new features, and the major version for breaking changes (before 1.0, breaking changes bump the minor version).

1. In `CHANGELOG.md`, rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`, add a new empty `## [Unreleased]` section above it, and update the comparison links at the bottom of the file.
2. Set the version with `poetry version X.Y.Z` (or `poetry version patch`, `minor` or `major`).
3. Commit both files as `Release X.Y.Z` and get the commit onto `main`.
4. On GitHub, draft a release with a new tag `vX.Y.Z` on that commit, paste the changelog section as the release notes, and publish it. The release workflow checks that the tag matches the version, builds the package and uploads it to PyPI.

