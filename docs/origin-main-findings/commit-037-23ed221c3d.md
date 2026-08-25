# Origin #37 closure: CI speed

Upstream commit `23ed221c3d` reduces Python CI wall-clock time by:

- collecting coverage only on the Linux 3.12 leg in
  `.github/workflows/run-unit-tests.yml`
- switching several pytest-xdist invocations to `--dist=worksteal` in
  `Makefile`
- caching the Python optional-dependency AST scan in
  `tests/harness/optional_deps.py`
- adding focused Python regression coverage in
  `tests/unit/harness/test_optional_deps.py`

The diff does not touch `rust/`, the native CLI surface, the runtime hot path,
or any Rust-launched integration seam. Native behavior is therefore unchanged,
so there is no Rust implementation to port and no Rust TDD target to add.

Disposition: not-applicable; exact merge performed for campaign ancestry.

Verification inventory:

- `pytest -q tests/unit/harness/test_optional_deps.py` using the existing shared
  project virtualenv: 4 passed.
- No Rust test candidate exists because the upstream change is limited to Python
  CI orchestration and Python test harness caching.

Graham review outcome:

- No findings. The merge changes only Python and GitHub Actions files, outside
  the Rust/runtime review hot paths governed by the Graham rubric.
