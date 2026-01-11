# Versioning & Releases

This project uses **SemVer-style** versioning (`MAJOR.MINOR.PATCH`) with a pragmatic emphasis on API stability for
the `rlm` package.

## What changes bump what?

- **PATCH** (`0.1.X`): bug fixes, test/docs improvements, internal refactors that do not change the public API.
- **MINOR** (`0.X.0`): new features or new adapters/flags that are backwards-compatible.
- **MAJOR** (`X.0.0`): breaking changes to the public API or runtime behavior.

## Release checklist

- Update `pyproject.toml` version.
- Update `CHANGELOG.md`.
- Run quality gates:
  - `uv run --group dev ruff format --check .`
  - `uv run --group dev ruff check .`
  - `uv run --group test pytest -m unit`
  - `uv run --group test pytest -m integration`
  - `uv run --group test pytest -m e2e` (Docker tests skip cleanly if unavailable)
  - `uv run --group test pytest -m packaging`
  - `uv run --group test pytest -m performance` (opt-in regression guard)
- Build artifacts:
  - `uv build --wheel --sdist`
- Tag the release:
  - `git tag vX.Y.Z`
  - `git push --tags`
