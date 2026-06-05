# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Subprocess entry point for running isolated benchmark iterations.

This module provides the entry point for running a single benchmark in a subprocess.
It's used by MultiRunOrchestrator to execute each run in complete isolation,
allowing the SystemController to call os._exit() without affecting the orchestrator.
"""

import os
import sys
from pathlib import Path

import orjson

from aiperf.common.constants import IS_WINDOWS

# Parent passes the api_key through this env var rather than writing it
# into run_config.json (which is redacted by EndpointConfig's
# field_serializer). We pop+restore in main() so neither child processes
# nor any logging path inherits the plaintext value.
_INJECTED_API_KEY_ENV = "AIPERF_INJECTED_API_KEY"


def _release_inherited_pipes_on_windows() -> None:
    """Release inherited stdio pipes on Windows so this intermediate
    sweep-iteration process can shut down cleanly. No-op on POSIX.

    Sweep iterations are spawned via subprocess.run with stdout=sys.stdout
    inherited from the orchestrator master, which on Windows propagates
    pytest's subprocess.PIPE all the way down. The iteration's own grandchild
    workers already redirect via the bootstrap fix, but the iteration process
    itself still holds the inherited pipe handle, so its ``os._exit()`` can
    hang or segfault during ``DLL_PROCESS_DETACH``.

    Delegates to ``bootstrap._redirect_stdio_to_devnull`` so the per-process
    stderr-to-file pattern (with 0o600 hardening and atexit-cleanup) is
    applied symmetrically — discarding stderr here would lose tracebacks
    from iteration-process crashes during sweep-mode benchmarking on
    Windows. See bootstrap.py::_redirect_stdio_to_devnull for the full
    rationale.
    """
    if not IS_WINDOWS:
        return
    # Late import to avoid circular load: bootstrap imports from many other
    # subsystems; subprocess_runner is loaded before bootstrap completes its
    # own imports in some test paths.
    from aiperf.common.bootstrap import _redirect_stdio_to_devnull

    _redirect_stdio_to_devnull()


def main() -> None:
    """Run a single benchmark from a BenchmarkRun JSON file.

    Usage:
        python -m aiperf.orchestrator.subprocess_runner /path/to/run_config.json
    """
    if len(sys.argv) != 2:
        print(
            "Usage: python -m aiperf.orchestrator.subprocess_runner <run_config.json>",
            file=sys.stderr,
        )
        sys.exit(1)

    config_file = Path(sys.argv[1])

    if not config_file.exists():
        print(f"Error: Config file not found: {config_file}", file=sys.stderr)
        sys.exit(1)

    # Pop (don't just read) so child processes the benchmark spawns
    # don't inherit the secret. Restore onto the loaded config below.
    injected_api_key = os.environ.pop(_INJECTED_API_KEY_ENV, None)

    from aiperf.cli_runner import _run_single_benchmark
    from aiperf.config import BenchmarkRun

    try:
        with open(config_file, "rb") as f:
            data = orjson.loads(f.read())

        run = BenchmarkRun.model_validate(data)
        if injected_api_key is not None:
            run.cfg.endpoint.api_key = injected_api_key
        _run_single_benchmark(run)

    except KeyError as e:
        print(f"Error: Missing required config key: {e}", file=sys.stderr)
        sys.exit(1)
    except orjson.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:  # noqa: BLE001 - subprocess entry point: final safety net so the parent orchestrator gets a nonzero exit + traceback rather than an opaque crash
        print(f"Error: Failed to run benchmark: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Release inherited pipe handles only when actually run as a subprocess
    # (`python -m aiperf.orchestrator.subprocess_runner ...`). Calling
    # ``main()`` from unit tests must NOT redirect stderr — pytest's capsys
    # needs to see the error prints, and there are no inherited pipes to
    # release in an in-process call. ``_release_inherited_pipes_on_windows``
    # itself is also gated on IS_WINDOWS so this is belt-and-suspenders.
    _release_inherited_pipes_on_windows()
    main()
