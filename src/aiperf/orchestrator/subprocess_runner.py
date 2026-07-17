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

# Sensitive entries from EndpointConfig.headers (Authorization, X-API-Key, etc.)
# are forwarded via this env var for the same reason: the headers serializer
# replaces those values with "<redacted>" on every JSON dump, so the on-disk
# run_config.json is secret-free but the child needs the real values to talk
# to the upstream. Non-sensitive headers round-trip through run_config.json
# normally and are not duplicated here.
_INJECTED_HEADERS_ENV = "AIPERF_INJECTED_HEADERS"

# Endpoint URLs that carry userinfo (``user:pass@host``) are forwarded via
# this env var. ``EndpointConfig.urls`` has an unconditional _redact_urls
# serializer (no when_used="json" guard), so even non-JSON dumps strip
# userinfo. The parent sets this only when at least one URL would be
# redacted, so plain http(s)://host URLs never round-trip through env vars.
_INJECTED_ENDPOINT_URLS_ENV = "AIPERF_INJECTED_ENDPOINT_URLS"


def _parse_injected_dict(name: str, raw: str | None) -> dict[str, str] | None:
    """Decode a JSON-object IPC env-var payload, validating shape.

    Used for ``AIPERF_INJECTED_HEADERS``. Returns ``None`` when ``raw`` is
    unset. Raises ``ValueError`` on malformed JSON or shape mismatch — a
    non-object payload, or non-string values that would bypass the
    ``EndpointConfig.headers`` ``dict[str, str]`` constraint once
    ``.update()``-ed onto the loaded config (``.update`` mutates in place and
    never re-runs field validation). Decode errors are re-raised as
    ``ValueError`` so a bad env var surfaces via main()'s structured error
    envelope with the env-var name, never misattributed to the config file.
    """
    if not raw:
        return None
    try:
        decoded = orjson.loads(raw)
    except orjson.JSONDecodeError as e:
        raise ValueError(f"{name} contains invalid JSON: {e}") from e
    if not isinstance(decoded, dict):
        raise ValueError(
            f"{name} must decode to a JSON object, got {type(decoded).__name__}"
        )
    if not all(isinstance(v, str) for v in decoded.values()):
        raise ValueError(f"{name} must decode to a JSON object with string values")
    return decoded


def _parse_injected_str_list(name: str, raw: str | None) -> list[str] | None:
    """Decode a JSON-list-of-strings IPC env-var payload, validating shape.

    Used for ``AIPERF_INJECTED_ENDPOINT_URLS``. Returns ``None`` when
    ``raw`` is unset. Raises ``ValueError`` on malformed JSON or shape
    mismatch; decode errors are re-raised as ``ValueError`` so a bad env
    var is never misattributed to the config file.
    """
    if not raw:
        return None
    try:
        decoded = orjson.loads(raw)
    except orjson.JSONDecodeError as e:
        raise ValueError(f"{name} contains invalid JSON: {e}") from e
    if not isinstance(decoded, list) or not all(isinstance(u, str) for u in decoded):
        raise ValueError(f"{name} must decode to a JSON list of strings")
    return decoded


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
    # Parsing of the JSON-encoded vars is deferred into the try block so
    # malformed payloads surface via the structured error envelope rather
    # than an unguarded JSONDecodeError.
    injected_api_key = os.environ.pop(_INJECTED_API_KEY_ENV, None)
    injected_headers_raw = os.environ.pop(_INJECTED_HEADERS_ENV, None)
    injected_urls_raw = os.environ.pop(_INJECTED_ENDPOINT_URLS_ENV, None)

    from aiperf.cli_runner import _run_single_benchmark
    from aiperf.config import BenchmarkRun

    try:
        injected_headers = _parse_injected_dict(
            _INJECTED_HEADERS_ENV, injected_headers_raw
        )
        injected_urls = _parse_injected_str_list(
            _INJECTED_ENDPOINT_URLS_ENV, injected_urls_raw
        )

        with open(config_file, "rb") as f:
            data = orjson.loads(f.read())

        run = BenchmarkRun.model_validate(data)
        if injected_api_key is not None:
            run.cfg.endpoint.api_key = injected_api_key
        if injected_headers:
            run.cfg.endpoint.headers.update(injected_headers)
        if injected_urls:
            run.cfg.endpoint.urls = injected_urls
        _run_single_benchmark(run)

    except KeyError as e:
        print(f"Error: Missing required config key: {e}", file=sys.stderr)
        sys.exit(1)
    except orjson.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:  # subprocess entry point: final safety net so the parent orchestrator gets a nonzero exit + traceback rather than an opaque crash
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
