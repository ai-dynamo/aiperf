# Fern Docs: Local Strict Relative-Path / Broken-Link Detection

## Problem

CI (`.github/workflows/fern-docs.yml`) validates documentation with three Fern
commands run against the **md_to_mdx-transformed** content:

- `fern check`
- `fern check --warnings --strict-broken-links`
- `fern docs broken-links`

The local equivalent (`make test-fern-docs` -> `tests/unit/fern/test_fern_docs.py`)
only runs plain `fern check` plus a dev-server smoke test, and it runs them
against the implicit repo root (raw `docs/`). The strict broken-link / relative
-path detection that CI performs has no local counterpart, so a broken relative
link only surfaces in CI.

Running the strict checks directly against the raw source `docs/` does **not**
work: raw GitHub markdown contains HTML comments (`<!-- ... -->`) that Fern's
MDX parser rejects, causing the `valid-markdown-links` rule to fail to
initialize (a false failure unrelated to any real link). CI avoids this by
converting markdown to MDX via `fern/md_to_mdx.py` before checking. The local
tests must do the same.

## Goal

Mirror CI's strict checks locally so relative/broken-link failures fail
`make test-fern-docs` before a push, validating the same **converted** content
CI publishes.

## Design

All changes live in `tests/unit/fern/test_fern_docs.py`.

### Shared staged-conversion fixture

Add a session-scoped `staged_fern_docs` fixture that reproduces the
`make fern-preview` / CI staging pipeline:

1. Locate the repo root via `Path(__file__).parents[3]`.
2. Copy `fern/` (excluding `.local-preview`) and `docs/` into a
   `tmp_path_factory` tree as `<tmp>/fern` and `<tmp>/docs`.
3. Run `python3 fern/md_to_mdx.py --dir <tmp>/docs` to convert HTML comments and
   GitHub callouts to MDX.
4. Yield the staged `<tmp>/fern` directory.

The unchanged `path: ../docs/index.yml` in `fern/docs.yml` resolves correctly in
this layout (verified manually), so no `docs.yml` transformation is needed —
unlike CI, which transforms paths only because docs-website uses a different
directory layout.

### Tests

- `test_fern_check` (existing): run with `cwd=<staged fern dir>` instead of the
  implicit repo root, validating converted content.
- `test_fern_docs_dev_starts` (existing): same — run from the staged dir.
- `test_fern_check_strict` (new): run `fern check --warnings
  --strict-broken-links` in the staged dir; assert `returncode == 0`. The only
  remaining output is the auth-skipped-redirects and accent-contrast warnings,
  which are not errors under this flag; broken/relative links are. Surface
  stdout/stderr on failure.
- `test_fern_broken_links` (new): run `fern docs broken-links` in the staged
  dir; assert `returncode == 0`, surfacing stdout/stderr on failure.

All tests keep the existing `pytestmark` skip when the `fern` CLI is absent.

### Makefile

No target change — `make test-fern-docs` already runs the whole
`tests/unit/fern/` suite. Update the target's one-line comment to mention the
broken-links check alongside the strict check it already advertises.

## Testing

- `make test-fern-docs` (or `pytest tests/unit/fern/ -m fern`) passes locally
  with `fern` installed, exercising all four checks against converted content.
- Manually verified: against converted content the strict check reports 0
  errors and `fern docs broken-links` reports "All checks passed"; against raw
  `docs/` the strict check falsely errors on HTML-comment parsing.

## Out of Scope

- New custom relative-path linter independent of Fern.
- Changes to CI, `md_to_mdx.py`, or the conversion logic itself.
