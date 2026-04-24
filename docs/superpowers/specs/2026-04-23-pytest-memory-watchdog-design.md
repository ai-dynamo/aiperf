# Pytest Memory Watchdog — Design (v2)

**Date:** 2026-04-23
**Status:** Proposed (replaces `2026-04-23-pytest-memory-guard-design.md`)
**Scope:** `tests/unit/` and `tests/component_integration/` (in-process suites)

## Context

The RLIMIT_AS approach (v1 spec) was implemented, tested, and abandoned —
see the v1 spec's Post-Mortem for the failure table and root cause
(library mmap reservations and xdist worker accumulation). This v2 spec
replaces it with the "RSS watchdog" approach that the v1 Post-Mortem
recommended.

## Problem (unchanged)

A pytest in-process test that enters a runaway allocation loop grows
until the OS OOM-killer fires, killing the xdist worker with SIGKILL and
producing a cryptic "worker crashed" message. The existing `timeout = 60`
doesn't help when allocation is fast enough to exhaust RAM in seconds.
Goal: convert that mode into an attributable, diagnosable failure.

## Approach

A **single daemon thread per xdist worker** samples
`/proc/self/status` Rss every 500 ms. When the current test's RSS
exceeds its effective threshold, the watchdog:

1. Writes a diagnostic block to stderr: the current test's nodeid,
   observed RSS, configured threshold, and a py-spy-like snapshot of
   each Python thread's stack (via `sys._current_frames()` and
   `traceback.format_stack`).
2. Calls `os._exit(137)` to kill the worker immediately.

xdist treats this as a worker crash. The human reader sees the stderr
block and knows exactly which test was running, how much RSS it had
consumed, and what it was doing at kill time.

## Why RSS and not virtual

RSS measures actual resident pages. Library reservations (`torch`,
`pyarrow`, `cffi`, `soundfile`) show up in virtual size but not RSS.
A runaway `bytearray(N*1024*1024); append; repeat` grows RSS linearly.
This is exactly the failure mode we want to catch, and exactly the
difference that broke the v1 implementation.

## Why `os._exit(137)` and not an in-process exception

The v1 Post-Mortem weighed signal-based in-process attribution and
rejected it for v1 complexity reasons. Per-test attribution via
`os.kill(os.getpid(), SIGUSR1)` + handler is tempting but introduces
races between signal delivery and test body completion, and requires
installing signal handlers that may conflict with other plugins.
`os._exit(137)` is unambiguous: one worker down, one current test named
in stderr, no handler interactions. xdist continues with remaining
workers.

Trade-off: the failing test doesn't appear in pytest's failure list.
The user reads the stderr diagnostic to know which test killed the
worker. This is acceptable because (a) the runaway is the priority
signal, (b) the stderr attribution is always correct, (c) in steady
state no test should ever trigger this so UX polish is not worth
complexity.

## Marker and env-var semantics (unchanged from v1)

| Location / marker / env | Effective threshold |
| --- | --- |
| outside `tests/unit/` + `tests/component_integration/` | disabled |
| non-Linux platform | disabled |
| `AIPERF_TEST_MEMORY_LIMIT_MB=0` | disabled globally |
| `@pytest.mark.no_memory_limit` | disabled for that test |
| `@pytest.mark.memory_limit(mb=N)` | N MiB for that test |
| `AIPERF_TEST_MEMORY_LIMIT_MB=N` (N > 0) | N MiB default |
| otherwise | 8192 MiB (8 GiB) default |

Default 8 GiB because: (a) v1's 4 GiB trial hit legitimate limits
even on virtual; RSS of 8 GiB per worker is far above any normal test
but far below OOM-kill thresholds on dev boxes, (b) runaway loops
typically grow MiB/s and hit 8 GiB in seconds, well before 60 s
pytest-timeout.

## Component design

### `tests/conftest.py` additions (appended after existing code)

```python
import os
import sys
import threading
import time
import traceback

_RSS_STATUS_PATH = "/proc/self/status"
_WATCHDOG_SUPPORTED = sys.platform == "linux" and os.path.exists(_RSS_STATUS_PATH)

_DEFAULT_WATCHDOG_MB = 8192
_WATCHDOG_INTERVAL_S = 0.5
_WATCHDOG_ENV_VAR = "AIPERF_TEST_MEMORY_LIMIT_MB"
_WATCHDOG_PATH_PREFIXES = ("tests/unit/", "tests/component_integration/")

# Module-level state the hookwrapper updates and the watchdog thread reads.
# Single writer (main thread via hookwrapper), single reader (watchdog
# thread); atomic dict writes are sufficient — no lock needed.
_watchdog_state: dict = {
    "active": False,
    "threshold_bytes": _DEFAULT_WATCHDOG_MB * 1024 * 1024,
    "nodeid": None,
}

# Injection point for tests: override to capture calls instead of exiting.
_watchdog_kill_action: "Callable[[str, int, int], None]" = None  # set below
```

### Watchdog loop

```python
def _read_rss_bytes() -> int | None:
    try:
        with open(_RSS_STATUS_PATH) as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # "VmRSS:  12345 kB"
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _default_watchdog_kill(nodeid: str, rss_bytes: int, threshold_bytes: int) -> None:
    sys.stderr.write(
        f"\n=== pytest memory watchdog tripped ===\n"
        f"test:      {nodeid}\n"
        f"rss:       {rss_bytes // (1024 * 1024)} MiB\n"
        f"threshold: {threshold_bytes // (1024 * 1024)} MiB\n"
        f"action:    killing worker pid {os.getpid()} with exit code 137\n"
        f"--- python thread stacks ---\n"
    )
    for tid, frame in sys._current_frames().items():
        sys.stderr.write(f"\n[thread {tid}]\n")
        sys.stderr.write("".join(traceback.format_stack(frame)))
    sys.stderr.write("=== end pytest memory watchdog ===\n")
    sys.stderr.flush()
    os._exit(137)


_watchdog_kill_action = _default_watchdog_kill


def _watchdog_loop() -> None:
    while True:
        time.sleep(_WATCHDOG_INTERVAL_S)
        if not _watchdog_state["active"]:
            continue
        rss = _read_rss_bytes()
        if rss is None:
            continue
        threshold = _watchdog_state["threshold_bytes"]
        if rss > threshold:
            nodeid = _watchdog_state["nodeid"] or "<unknown>"
            # Deactivate before killing so that test overrides in tests
            # (which swap _watchdog_kill_action) don't re-trigger.
            _watchdog_state["active"] = False
            _watchdog_kill_action(nodeid, rss, threshold)
```

### Lifecycle hooks

```python
def pytest_configure(config):
    ...existing body...
    if _WATCHDOG_SUPPORTED:
        t = threading.Thread(target=_watchdog_loop, daemon=True, name="memory-watchdog")
        t.start()


def _in_guarded_suite(nodeid: str) -> bool:
    return any(nodeid.startswith(p) for p in _WATCHDOG_PATH_PREFIXES)


def _resolve_threshold_mb(item) -> int | None:
    if item.get_closest_marker("no_memory_limit") is not None:
        return None
    env_raw = os.environ.get(_WATCHDOG_ENV_VAR)
    env_mb: int | None = None
    if env_raw is not None:
        try:
            env_mb = int(env_raw)
        except ValueError:
            env_mb = None
        else:
            if env_mb == 0:
                return None
    marker = item.get_closest_marker("memory_limit")
    if marker is not None:
        mb = marker.kwargs.get("mb")
        if mb is None and marker.args:
            mb = marker.args[0]
        if isinstance(mb, int) and mb > 0:
            return mb
    if env_mb is not None and env_mb > 0:
        return env_mb
    return _DEFAULT_WATCHDOG_MB


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    if not _WATCHDOG_SUPPORTED or not _in_guarded_suite(item.nodeid):
        yield
        return
    threshold_mb = _resolve_threshold_mb(item)
    if threshold_mb is None:
        yield
        return
    _watchdog_state["threshold_bytes"] = threshold_mb * 1024 * 1024
    _watchdog_state["nodeid"] = item.nodeid
    _watchdog_state["active"] = True
    try:
        yield
    finally:
        _watchdog_state["active"] = False
        _watchdog_state["nodeid"] = None
```

## Testing

Three tests in `tests/unit/test_memory_watchdog.py`:

1. `test_watchdog_fires_when_rss_exceeds_threshold` — use a tight
   marker (`memory_limit(mb=64)`), replace `_watchdog_kill_action` with
   a lambda that records the call, allocate a 256 MiB `bytearray`,
   `time.sleep(1.5)` (three watchdog intervals), assert the kill action
   was called with correct nodeid and the observed RSS > threshold.
   Restore the kill action in a `finally`.

2. `test_no_memory_limit_marker_disables_watchdog` — marker
   `no_memory_limit`, replace kill action with a recorder, allocate 256
   MiB, sleep 1.5 s, assert the recorder was NOT called. Proves the
   marker deactivates the watchdog.

3. `test_default_threshold_applied` — no marker; after the hookwrapper
   setup runs (test body is already inside it), read
   `_watchdog_state["threshold_bytes"]` and assert it equals 8192 MiB.
   Proves the default is plumbed.

All three skip on non-Linux via a module-level
`pytest.mark.skipif(not _WATCHDOG_SUPPORTED, ...)`.

## Out of scope

- In-process attribution via SIGUSR1. Could be added later if worker
  kill proves too noisy.
- Watchdog for `tests/integration/` or `tests/kubernetes/`. Those
  suites spawn subprocesses where RSS is split across children; a
  per-worker watchdog wouldn't see the growth.
- Configurable sampling interval. 500 ms is hard-coded.

## Rollout

Land as a single commit on `ajc/k8s` (conftest additions + three
tests). No Three-File Sync doc update required — this is test infra
only. The `memory_limit` / `no_memory_limit` markers already registered
in `pyproject.toml` from the v1 attempt are reused directly.
