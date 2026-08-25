Graham review for origin #37 (`23ed221c3d`)
Date: 2026-08-25

Scope reviewed:

- `.github/workflows/run-unit-tests.yml`
- `Makefile`
- `tests/harness/optional_deps.py`
- `tests/unit/harness/test_optional_deps.py`

Outcome:

- No findings.

Rationale:

- The merge contains no Rust changes.
- No request/token hot path, async runtime, transport, scheduler, graph, or
  metrics-core Rust code is modified.
- The Python harness change is narrowly scoped to CI/test collection caching and
  its focused regression tests pass.
