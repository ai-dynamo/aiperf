# Commit 061 — `4edbbf39d3`

## Upstream intent

The constrained qLogNEI path constructs a multi-output `SingleTaskGP`: its
training target contains the objective plus one column for every SLA filter.
The custom DSP Matern kernel must therefore use the GP's augmented batch shape
for both the outer scale and inner Matern kernels. Without that shape, BoTorch
fails while fitting a constrained model because its kernel parameters have only
one batch entry.

## Native comparison

`rust/cli/src/pyopt.rs` owns the native CLI's `search-pyo3` Optuna bridge. For
the `bo` sampler it imports and executes
`aiperf.orchestrator.search_planner._optuna_helpers.build_qlognei_candidates_func`
from this repository. The imported function owns both `SingleTaskGP`
construction and the DSP kernel call. Consequently this is product behavior of
the Rust CLI even though the implementation is Python, and it requires the
upstream batch-shape propagation plus its focused BoTorch tests.

No separate Rust GP exists: `bayes.rs` deliberately owns only ask/tell and
convergence while optional BoTorch fitting remains inside the embedded CPython
process. Duplicating a GP kernel in Rust would add an unused second
implementation and would not affect the executing `bo` path.

## Port boundary

Port the exact helper behavior and focused Python regression coverage. Verify
the normal test suite when the optional BoTorch stack is available, and verify
the Rust-owned seam separately with its existing feature-gated library tests.
The upstream workflow-only BoTorch CI job is not a native behavior surface.

## Implementation and verification

`make_dsp_kernel` now accepts an optional `batch_shape` and forwards it to the
inner `MaternKernel` and outer `ScaleKernel`. The qLogNEI factory obtains
`aug_batch_shape` from `SingleTaskGP.get_batch_dimensions` after building the
objective-plus-constraints target, so single-output and constrained paths both
use BoTorch's own shape rules. The test spy is a `SingleTaskGP` subclass rather
than a function, preserving the classmethod used by the production path.

TDD red run, after installing the optional CPU BoTorch stack, reported four
expected `TypeError` failures because `make_dsp_kernel` did not accept
`batch_shape`. Green verification used the isolated worktree source:

- `pytest -p no:rerunfailures -q tests/unit/orchestrator/search_planner/test_botorch_kernel.py`: 7 passed.
- `pytest -p no:rerunfailures -q -m slow tests/unit/orchestrator/search_planner/test_optuna_dsp_kernel.py::test_qlognei_candidates_func_fits_dsp_kernel`: 1 passed.
- the new one- and two-SLA constrained qLogNEI cases: 1 passed each.
- `cargo test -p aiperf-cli --features search-pyo3 --lib`: 281 passed with
  `CARGO_TARGET_DIR=/mnt/4tb/aiperf-origin-port-061-target` and `/usr/bin/sccache`.

The initial Rust test invocation compiled successfully but needed the active
Python `LIBDIR` prepended to `LD_LIBRARY_PATH` for the Pyo3 test executable to
find `libpython3.13.so.1.0`; the rerun passed. It emitted four unrelated
pre-existing runtime warnings.

## Graham review

Reviewed the complete port range in two passes against the Graham Rust rubric.
There are no Rust hot-path changes, no added synchronization, blocking work,
logging, or production `unwrap`/`expect`; the Python helper only forwards the
shape calculated by BoTorch. The test subclass preserves classmethod behavior
without altering production execution. No Critical, Important, or Minor
findings.
