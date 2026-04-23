# Pytest Memory Guard — Design

**Date:** 2026-04-23
**Status:** Proposed
**Scope:** `tests/unit/` and `tests/component_integration/` (in-process suites)

## Problem

The aiperf pytest configuration already caps wall-clock runtime at 60 s via
`pytest-timeout` (`timeout = 60`, `timeout_method = "thread"`). This catches
tests that hang on I/O or sleep loops. It does **not** catch tests that enter
a tight allocation loop and grow memory without bound — a realistic failure
mode for LLM-authored code (e.g. an async generator that never releases its
backlog, or a list being `.append`-ed inside an accidental infinite `while`).

When that happens under `pytest-xdist -n auto`, the worker process exhausts
RAM, the kernel OOM-killer fires, and the worker dies with SIGKILL. Pytest
then reports a cryptic "worker crashed" / "internalerror" message rather
than pointing at the test that caused it. Other workers may also be killed
as collateral. The goal is to convert this failure mode into a clean,
attributable `MemoryError` in the offending test frame.

## Goals

1. Any runaway-allocation test in `tests/unit/` or
   `tests/component_integration/` fails as a standard pytest failure, with a
   traceback pointing at the offending allocation, before the OS OOM-killer
   is involved.
2. The cap is cheap enough to be always-on (no measurable overhead per
   test).
3. Legitimately-heavy tests can opt out or raise their cap via a marker.
4. Local-debug and unusual-machine workflows can disable the cap via an
   environment variable.
5. Zero impact on integration, k8s, e2e, server_unit, performance, stress,
   and ffmpeg suites (their failure modes are elsewhere; subprocesses they
   spawn aren't bounded by an in-process rlimit anyway).

## Non-Goals

- RSS watchdog thread / tracemalloc integration. The cap is a safety net,
  not a profiler.
- cgroup wrappers. They would cover child processes but replace
  `MemoryError` with SIGKILL, which is exactly the failure mode we are
  trying to eliminate.
- Retroactively tagging existing tests. The audit (see below) shows none
  need tagging today; add markers reactively if the default starts
  tripping.
- Covering subprocess children spawned by integration tests. Out of scope.

## Approach

Use `resource.setrlimit(RLIMIT_AS, (cap, original_hard))` to cap the
**virtual address space** of each pytest (xdist worker) process, applied
per-test via an autouse fixture and restored on teardown.

`RLIMIT_AS` (not `RLIMIT_DATA` or `RLIMIT_RSS`) because:

- It is honored by `mmap`, `brk`, and thread-stack allocations — all the
  paths a Python allocation loop actually hits.
- Exceeding it causes the next allocation attempt to fail in userspace with
  `MemoryError` raised in the current Python frame, which pytest captures
  as a normal test failure.
- `RLIMIT_DATA` misses anonymous mmaps (which CPython and glibc use for
  large allocations on 64-bit Linux). `RLIMIT_RSS` is advisory / ignored on
  modern Linux.

Default cap: **4 GiB** of virtual address space per worker process.

## Component Design

### 1. `tests/conftest.py` additions

A new module-level helper and an autouse fixture.

```python
# Platform detection + initial-limit capture at import time.
try:
    import resource
    _RLIMIT_AS_SUPPORTED = hasattr(resource, "RLIMIT_AS")
except ImportError:  # Windows
    _RLIMIT_AS_SUPPORTED = False

_DEFAULT_CAP_MB = 4096  # 4 GiB
_ENV_VAR = "AIPERF_TEST_MEMORY_LIMIT_MB"

_TARGET_PATH_PREFIXES = ("tests/unit/", "tests/component_integration/")
```

The fixture body is purely function-scoped (autouse=True). It skips the
cap entirely when any of these is true:

- The platform lacks `RLIMIT_AS` (Windows, possibly macOS — detected at
  import time via `hasattr(resource, "RLIMIT_AS")`, not hardcoded per
  platform).
- The test's node path does not start with one of
  `_TARGET_PATH_PREFIXES`.
- `@pytest.mark.no_memory_limit` is present on the test.
- Env var `AIPERF_TEST_MEMORY_LIMIT_MB=0` is set.

Effective cap resolution order (first match wins):

1. `@pytest.mark.no_memory_limit` → skip entirely (beats everything,
   including `memory_limit` if both are somehow applied).
2. `AIPERF_TEST_MEMORY_LIMIT_MB=0` → skip entirely.
3. `@pytest.mark.memory_limit(mb=N)` → N MiB (beats env-var override).
4. `AIPERF_TEST_MEMORY_LIMIT_MB=N` (N > 0) → N MiB.
5. `_DEFAULT_CAP_MB` → 4096 MiB.

On entry: record original `(soft, hard)`, call
`setrlimit(RLIMIT_AS, (cap_bytes, hard))`. On exit (`finally`): restore
the original pair. Teardown restoration is important because a subsequent
test with `no_memory_limit` would otherwise inherit a stale cap.

### 2. Marker registration (`pyproject.toml`)

Add two markers to the existing `markers = [...]` list:

```
"memory_limit(mb=N): cap this test's virtual address space at N MiB (default 4096)",
"no_memory_limit: disable the per-test virtual address space cap entirely",
```

### 3. Documentation

Add a short subsection under `## Testing Conventions` in `CLAUDE.md`
(plus the two sync files `.github/copilot-instructions.md` and
`.cursor/rules/python.mdc`) describing:

- The default 4 GiB per-worker cap on unit / component_integration suites.
- When to use `memory_limit(mb=N)` vs `no_memory_limit`.
- The `AIPERF_TEST_MEMORY_LIMIT_MB` env-var escape hatch.

Nothing in `docs/dev/patterns.md` needs to change — this is test infra,
not a code pattern.

## Data / Decision Table

| Test location                              | Marker                         | Env var                                 | Effective cap |
| ------------------------------------------ | ------------------------------ | --------------------------------------- | ------------- |
| unit or component_integration              | none                           | unset                                   | 4 GiB         |
| unit or component_integration              | `memory_limit(mb=N)`           | any                                     | N MiB         |
| unit or component_integration              | `no_memory_limit`              | any                                     | unbounded     |
| unit or component_integration              | none                           | `AIPERF_TEST_MEMORY_LIMIT_MB=0`         | unbounded     |
| unit or component_integration              | none                           | `AIPERF_TEST_MEMORY_LIMIT_MB=N` (N > 0) | N MiB         |
| integration / k8s / e2e / server_unit / …  | any                            | any                                     | unbounded (skip) |
| any platform without `RLIMIT_AS` (Windows) | any                            | any                                     | unbounded (skip) |

## Failure UX

Before: runaway allocation → worker RAM exhausted → OS OOM-killer →
SIGKILL → xdist reports "worker <id> crashed" → user cannot tell which
test was at fault.

After: runaway allocation → `malloc` / `mmap` returns NULL at the cap →
CPython raises `MemoryError` in the current Python frame → pytest reports
a normal `FAILED` with a full traceback pointing at the offending line.
The existing `timeout = 60` still runs in parallel; whichever trips first
wins. (In practice `MemoryError` trips in seconds for tight loops, well
before the 60 s timeout.)

## Audit Findings

A grep-driven audit of `tests/unit/` and `tests/component_integration/`
for legitimate large-memory workloads was performed on 2026-04-23. All
hits are well below 4 GiB:

- Largest real buffers: ~1 MB payloads in
  `tests/unit/operator/test_download_response.py:225` and
  `tests/unit/transports/test_aiohttp_client.py:422`.
- `tests/unit/server_metrics/helpers.py:207` uses `np.zeros((N_snapshots,
  N_buckets))` with both dimensions on the order of 10s — kilobytes.
- Large integer literals (`1_000_000`, `100_000_000`, etc.) in
  timing/credit/trace tests are nanosecond timestamps and rate configs,
  not allocations.
- `tests/unit/conftest.py` replaces the HF `Tokenizer.from_pretrained`
  with a `FakeTokenizer` fixture; no real weights ever load under unit.
- Existing `@pytest.mark.stress`, `@pytest.mark.slow`,
  `@pytest.mark.performance` tests in scope are wall-clock / concurrency
  heavy, not memory heavy.

Conclusion: ship the 4 GiB default with zero pre-emptive markers. Add
`memory_limit(mb=N)` reactively if a future fixture (e.g. a large
parquet) legitimately needs more.

## Testing

New file `tests/unit/test_memory_limit_guard.py` with four tests:

1. `test_default_cap_triggers_memory_error` — with
   `@pytest.mark.memory_limit(mb=256)`, allocate `bytearray(32 * 1024 *
   1024)` in a loop; expect `MemoryError` in well under 5 s. Verifies
   the cap fires on a real allocation loop.
2. `test_marker_override_raises_cap` — with
   `@pytest.mark.memory_limit(mb=1024)`, allocate a single 300 MiB
   bytearray successfully. Verifies marker overrides the 256 MiB case
   above (regression against off-by-one logic).
3. `test_no_memory_limit_marker_disables_guard` — with
   `@pytest.mark.no_memory_limit` and
   `AIPERF_TEST_MEMORY_LIMIT_MB=128` set via `monkeypatch.setenv`,
   allocate a 300 MiB bytearray successfully. Verifies the opt-out marker
   beats the env var.
4. `test_cap_restored_after_test` — verifies teardown restores the
   original limit. Implemented as a pair: test 4a reads
   `resource.getrlimit(RLIMIT_AS)[0]` inside its own body and asserts it
   equals the active cap (4 GiB). Test 4b is decorated with
   `@pytest.mark.no_memory_limit`; its body reads the same value and
   asserts it equals the original soft limit captured at import time
   (via a module-level constant the test imports from `tests/conftest.py`).
   If teardown of 4a didn't restore, 4b's assertion fails.

All four tests live under `tests/unit/`, so they exercise the guard
itself.

On Windows CI (if ever added), all four tests should `pytest.skip` via a
skipif on `_RLIMIT_AS_SUPPORTED`.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| A CPython / glibc internal mmap reservation exceeds 4 GiB at import time. | Audit shows no test imports anywhere near this; if one appears, raise the default or add `memory_limit` marker. |
| Parallel threads in the test allocate concurrently and the `MemoryError` surfaces in a non-test thread (e.g. a worker from `asyncio.to_thread`). | Pytest captures uncaught exceptions in threads as warnings; the test's main frame will still trip on its next allocation. Acceptable — worst case is a slightly less precise traceback. |
| A test forks a subprocess that allocates memory. | `RLIMIT_AS` is per-process, so the child has its own copy of the limit (inherited). If the child OOMs it fails with a non-zero exit code, not SIGKILL — strictly better than today. |
| Developer runs a single test from IDE and hits the cap on a beefy machine that could easily handle more. | Document `AIPERF_TEST_MEMORY_LIMIT_MB=0` escape hatch prominently. |
| `resource.setrlimit` fails (e.g. tightened sandbox). | Wrap the call in `try/except (OSError, ValueError)` and log a warning; tests still run, just unbounded. Do not fail the test for an rlimit setup failure — the guard is best-effort. |

## Out of Scope

- Capping integration / k8s / e2e / server_unit suites. Their failure
  modes are in subprocesses or external clusters; an in-process rlimit
  wouldn't help.
- RSS monitoring, tracemalloc snapshots, leak detection across tests.
- Enforcing the guard in pre-commit. The hook runs only staged files and
  would duplicate CI.
- Windows support for the cap itself (detection-and-skip is sufficient).

## Rollout

1. Land the fixture + marker registration + documentation in one PR on
   `ajc/k8s`.
2. Run full `pytest tests/unit/ -n auto` and
   `pytest tests/component_integration/ -n auto`; any unexpected
   `MemoryError` is a real find or a marker candidate.
3. Merge. If CI breaks on a tagged-heavy test, add
   `@pytest.mark.memory_limit(mb=N)` with N chosen by inspection.

## References

- `pyproject.toml` `[tool.pytest.ini_options]` — existing `timeout = 60`,
  `markers = [...]` list, and `addopts` default deselections.
- `tests/conftest.py` — shared fixtures; this is the insertion point.
- CPython `resource` module — `RLIMIT_AS`, `setrlimit`, `getrlimit`.
