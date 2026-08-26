# Commit 002 — `0883bd1aee`

## Closure finding

The upstream version `0.12.0` change updates Python package/mock-server metadata
and documentation only. The exact commit is an ancestor of current `HEAD`, and
the current Python package remains `0.12.0` in `pyproject.toml`. Native crate
versions are independently declared; no runtime behavior or Rust port is
required. Disposition: **not-applicable**.

## Verification

`git merge-base --is-ancestor 0883bd1aee HEAD` passed. `git diff --check`
passed. No implementation or test substitution is warranted.
