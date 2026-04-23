# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared test configuration and fixtures for all test types.

ONLY ADD FIXTURES HERE THAT ARE USED IN ALL TEST TYPES.
DO NOT ADD FIXTURES THAT ARE ONLY USED IN A SPECIFIC TEST TYPE.
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import pytest

_RSS_STATUS_PATH = "/proc/self/status"
_SMAPS_ROLLUP_PATH = "/proc/self/smaps_rollup"
# Prefer PSS (proportional set size) because aiperf runs under pytest-xdist
# with ~24 workers; RSS double-counts shared library pages across workers and
# inflates reported per-worker memory by hundreds of MiB. PSS charges each
# process only its proportional share of shared pages, giving an honest
# per-worker cost for both the watchdog trigger and the --memory-report
# summary. smaps_rollup was added in Linux 4.14 (2017); fall back to RSS on
# older kernels so the guard still works.
_WATCHDOG_SUPPORTED = sys.platform == "linux" and (
    os.path.exists(_SMAPS_ROLLUP_PATH) or os.path.exists(_RSS_STATUS_PATH)
)
_PSS_AVAILABLE = sys.platform == "linux" and os.path.exists(_SMAPS_ROLLUP_PATH)
_MEMORY_METRIC_NAME = "pss" if _PSS_AVAILABLE else "rss"

# Duplicate fd 2 at conftest import time, before pytest's per-test capture
# plugin can dup2 over it. The watchdog writes its diagnostic here so the
# message survives pytest's capture and os._exit(137)'s capture-teardown
# bypass. Closed? We leak intentionally for the life of the process.
try:
    _WATCHDOG_STDERR_FD = os.dup(2)
except OSError:
    _WATCHDOG_STDERR_FD = 2

_DEFAULT_WATCHDOG_MB = 2048
_WATCHDOG_INTERVAL_S = 0.5
_WATCHDOG_ENV_VAR = "AIPERF_TEST_MEMORY_LIMIT_MB"
_WATCHDOG_PATH_PREFIXES = ("tests/unit/", "tests/component_integration/")

# Module-level state the hookwrapper updates and the watchdog thread reads.
# Single writer (main thread via hookwrapper), single reader (watchdog thread);
# atomic dict writes are sufficient - no lock needed.
_watchdog_state: dict = {
    "active": False,
    "threshold_bytes": _DEFAULT_WATCHDOG_MB * 1024 * 1024,
    "nodeid": None,
}

# Injection point for tests: override to capture calls instead of exiting.
# Exposed as a module attribute so tests can monkeypatch it cleanly.
_watchdog_kill_action: Callable[[str, int, int], None] | None = None

# Per-test RSS tracker for --memory-report. nodeid -> {start, peak, end} in
# bytes. Populated by the hookwrapper and watchdog loop; summarized in
# pytest_terminal_summary.
_per_test_rss: dict[str, dict[str, int | None]] = {}

# Path prefix -> markers to auto-enable (remove from default exclusions).
# When a user targets a path starting with the prefix, the listed markers are
# stripped from the ``-m 'not X and not Y ...'`` expression in addopts so the
# tests actually run instead of being silently deselected.
# Matching is bidirectional: targeting ``tests/kubernetes/gpu/vllm`` enables
# markers up the tree (k8s, gpu), and targeting ``tests/kubernetes`` enables
# all descendant markers (gpu, vllm, dynamo).
# Each entry only needs its own markers.
_PATH_MARKER_MAP: list[tuple[str, list[str]]] = [
    ("tests/kubernetes/gpu/vllm", ["vllm"]),
    ("tests/kubernetes/gpu/dynamo", ["dynamo"]),
    ("tests/kubernetes/gpu", ["gpu"]),
    ("tests/kubernetes", ["k8s"]),
    ("tests/integration", ["integration"]),
    ("tests/component_integration", ["component_integration"]),
]


def pytest_configure(config: pytest.Config) -> None:
    """Auto-enable markers when the user targets a specific test path, and
    start the per-worker memory watchdog thread.

    ``addopts`` in pyproject.toml excludes heavy test suites by default via
    ``-m 'not k8s and not gpu and ...'``.  When the user explicitly runs
    ``pytest tests/kubernetes/`` (or any other excluded path), this hook
    detects the target and strips the corresponding ``not <marker>`` clauses
    so the tests are collected instead of silently skipped.
    """
    _apply_path_marker_expansion(config)

    if _WATCHDOG_SUPPORTED:
        t = threading.Thread(target=_watchdog_loop, daemon=True, name="memory-watchdog")
        t.start()


def _apply_path_marker_expansion(config: pytest.Config) -> None:
    markexpr = getattr(config.option, "markexpr", "") or ""
    if not markexpr:
        return

    raw_args = [str(a) for a in config.invocation_params.args]
    if not raw_args:
        return

    # Normalize args to project-relative paths (handles absolute paths too)
    rootpath = config.invocation_params.dir
    rel_args: list[str] = []
    for arg in raw_args:
        # Strip ::TestClass::test_method node ids for path matching
        path_part = arg.split("::")[0]
        try:
            rel_args.append(str(Path(path_part).resolve().relative_to(rootpath)))
        except (ValueError, OSError):
            rel_args.append(path_part)

    # Collect all markers to enable based on targeted paths (bidirectional).
    # "tests/kubernetes/gpu/vllm/test_foo.py" starts with "tests/kubernetes"
    # so the k8s marker is enabled.  "tests/kubernetes" is a prefix of
    # "tests/kubernetes/gpu", so gpu is also enabled.
    enable: set[str] = set()
    for path_prefix, markers in _PATH_MARKER_MAP:
        if any(
            a.startswith(path_prefix) or path_prefix.startswith(a) for a in rel_args
        ):
            enable.update(markers)

    if not enable:
        return

    # Strip matching 'not <marker>' clauses from the expression
    exclude = {f"not {m}" for m in enable}
    parts = [p for p in re.split(r"\s+and\s+", markexpr) if p.strip() not in exclude]
    config.option.markexpr = " and ".join(parts) if parts else ""


def _read_memory_bytes() -> int | None:
    """Return per-process memory usage in bytes (PSS where available, else RSS).

    PSS (proportional set size) from /proc/self/smaps_rollup charges shared
    pages fractionally across mappers, giving an accurate per-worker cost
    under xdist. Falls back to VmRSS from /proc/self/status on kernels
    without smaps_rollup.
    """
    if _PSS_AVAILABLE:
        try:
            with open(_SMAPS_ROLLUP_PATH) as f:
                for line in f:
                    if line.startswith("Pss:"):
                        # "Pss:    12345 kB"
                        return int(line.split()[1]) * 1024
        except OSError:
            pass
    try:
        with open(_RSS_STATUS_PATH) as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # "VmRSS:  12345 kB"
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _default_watchdog_kill(nodeid: str, mem_bytes: int, threshold_bytes: int) -> None:
    # Build the diagnostic once.
    lines: list[str] = [
        "\n=== pytest memory watchdog tripped ===\n",
        f"test:      {nodeid}\n",
        f"{_MEMORY_METRIC_NAME}:       {mem_bytes // (1024 * 1024)} MiB\n",
        f"threshold: {threshold_bytes // (1024 * 1024)} MiB\n",
        f"action:    killing worker pid {os.getpid()} with exit code 137\n",
    ]
    log_path = os.environ.get(
        "AIPERF_WATCHDOG_LOG_FILE", f"/tmp/aiperf-pytest-watchdog-{os.getpid()}.log"
    )
    lines.append(f"log file:  {log_path}\n")
    lines.append("--- python thread stacks ---\n")
    for tid, frame in sys._current_frames().items():
        lines.append(f"\n[thread {tid}]\n")
        lines.append("".join(traceback.format_stack(frame)))
    lines.append("=== end pytest memory watchdog ===\n")
    blob = "".join(lines).encode("utf-8", "replace")

    # Write to a file so the diagnostic survives pytest's per-test fd-level
    # capture (which dup2s over fd 2 into a deleted tempfile) and
    # os._exit(137)'s capture-teardown bypass.
    try:
        with open(log_path, "wb") as f:
            f.write(blob)
    except OSError:
        pass
    # Best-effort: also try the saved fd in case the user is running without
    # pytest's fd capture (e.g. `-s`), so the diagnostic is visible inline.
    try:
        os.write(_WATCHDOG_STDERR_FD, blob)
    except OSError:
        pass
    os._exit(137)


_watchdog_kill_action = _default_watchdog_kill


def _watchdog_loop() -> None:
    while True:
        time.sleep(_WATCHDOG_INTERVAL_S)
        if not _watchdog_state["active"]:
            continue
        rss = _read_memory_bytes()
        if rss is None:
            continue
        nodeid = _watchdog_state["nodeid"] or "<unknown>"
        entry = _per_test_rss.get(nodeid)
        if entry is not None and (entry["peak"] is None or rss > entry["peak"]):
            entry["peak"] = rss
        threshold = _watchdog_state["threshold_bytes"]
        if rss > threshold:
            # Deactivate before killing so test overrides (which swap
            # _watchdog_kill_action) don't re-trigger on the next tick.
            _watchdog_state["active"] = False
            action = _watchdog_kill_action
            if action is not None:
                action(nodeid, rss, threshold)


def _in_guarded_suite(nodeid: str) -> bool:
    return any(nodeid.startswith(p) for p in _WATCHDOG_PATH_PREFIXES)


def _resolve_threshold_mb(item: pytest.Item) -> int | None:
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
def pytest_runtest_call(item: pytest.Item):
    if not _WATCHDOG_SUPPORTED or not _in_guarded_suite(item.nodeid):
        yield
        return
    threshold_mb = _resolve_threshold_mb(item)
    if threshold_mb is None:
        yield
        return
    mem_start = _read_memory_bytes()
    _per_test_rss[item.nodeid] = {
        "start": mem_start,
        "peak": mem_start,
        "end": None,
    }
    _watchdog_state["threshold_bytes"] = threshold_mb * 1024 * 1024
    _watchdog_state["nodeid"] = item.nodeid
    _watchdog_state["active"] = True
    try:
        yield
    finally:
        _watchdog_state["active"] = False
        _watchdog_state["nodeid"] = None
        mem_end = _read_memory_bytes()
        entry = _per_test_rss[item.nodeid]
        entry["end"] = mem_end
        if mem_end is not None and (entry["peak"] is None or mem_end > entry["peak"]):
            entry["peak"] = mem_end
        # Forward RSS data to the controller via user_properties so that
        # pytest_terminal_summary (which only runs in the controller under
        # xdist) can read it from the test report.
        item.user_properties.append(
            (
                "memory_rss",
                {
                    "start": entry["start"],
                    "peak": entry["peak"],
                    "end": entry["end"],
                },
            )
        )


# Controller-side aggregation of RSS data forwarded via user_properties.
_collected_rss: dict[str, dict[str, int | None]] = {}


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.when != "call":
        return
    for name, value in report.user_properties:
        if name == "memory_rss":
            _collected_rss[report.nodeid] = value
            break


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--memory-report",
        action="store_true",
        default=False,
        help=(
            "Print per-test peak RSS and start->end delta at end of session "
            "(guarded suites only: tests/unit/, tests/component_integration/)."
        ),
    )
    parser.addoption(
        "--memory-report-top",
        type=int,
        default=25,
        help="Number of tests to include in --memory-report (sorted by peak RSS). Default 25.",
    )


def pytest_terminal_summary(
    terminalreporter, exitstatus: int, config: pytest.Config
) -> None:
    if not config.getoption("--memory-report"):
        return
    if not _collected_rss:
        return
    top_n = config.getoption("--memory-report-top")
    rows = sorted(
        _collected_rss.items(),
        key=lambda kv: (kv[1]["peak"] or 0),
        reverse=True,
    )[:top_n]
    mib = 1024 * 1024
    terminalreporter.section(
        f"memory report (top {len(rows)} by peak {_MEMORY_METRIC_NAME.upper()})"
    )
    terminalreporter.write_line(
        f"  {'peak':>8}  {'delta':>8}  {'start':>8}  {'end':>8}  test"
    )
    for nodeid, entry in rows:
        start = entry["start"] or 0
        peak = entry["peak"] or 0
        end = entry["end"] or 0
        delta = (
            (end - start)
            if (entry["start"] is not None and entry["end"] is not None)
            else 0
        )
        terminalreporter.write_line(
            f"  {peak // mib:>5d}MiB  {delta // mib:>+5d}MiB  "
            f"{start // mib:>5d}MiB  {end // mib:>5d}MiB  {nodeid}"
        )
