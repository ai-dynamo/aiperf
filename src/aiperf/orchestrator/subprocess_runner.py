# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Subprocess entry point for running isolated benchmark iterations.

This module provides the entry point for running a single benchmark in a subprocess.
It's used by MultiRunOrchestrator to execute each run in complete isolation,
allowing the SystemController to call os._exit() without affecting the orchestrator.
"""

import sys
from pathlib import Path

import orjson

from aiperf.common.constants import IS_WINDOWS
from aiperf.common.endpoint_credentials import (
    apply_endpoint_credentials,
    consume_endpoint_credentials,
    credential_values,
    redact_credential_text,
)

# Endpoint credentials (api_key, sensitive headers, userinfo-bearing URLs) are
# redacted out of run_config.json by EndpointConfig's field serializers, so the
# parent hands the real values to this child through environment variables
# instead. aiperf.common.endpoint_credentials owns those variable names and the
# pop-validate-apply sequence; see consume_endpoint_credentials there.


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
    # don't inherit the secret -- this consumes OPENAI_API_KEY too, which the
    # parent has already resolved into endpoint.api_key. Restore onto the
    # loaded config below.
    # Parsing of the JSON-encoded vars is deferred into the try block so
    # malformed payloads surface via the structured error envelope rather
    # than an unguarded JSONDecodeError.
    from aiperf.cli_runner import _run_single_benchmark
    from aiperf.config import BenchmarkRun

    # Every credential this process rehydrates is scrubbed from error output
    # below: stderr of this subprocess is spliced verbatim into RunResult.error
    # and the orchestrator's logs by LocalSubprocessExecutor, so an exception
    # message or traceback frame that happens to echo the key (a connection
    # error naming a userinfo URL, a config repr, an HTTP client that formats
    # its auth header) would otherwise persist the secret. Redacting here --
    # at the only point that holds the plaintext -- keeps it off every
    # downstream path at once.
    secrets: list[str] = []
    try:
        credentials = consume_endpoint_credentials()
        secrets.extend(credential_values(credentials=credentials))

        with open(config_file, "rb") as f:
            data = orjson.loads(f.read())

        run = BenchmarkRun.model_validate(data)
        # require_resolved: a run_config.json replayed by hand (the documented
        # `python -m aiperf.orchestrator.subprocess_runner <file>` usage) still
        # carries the redaction placeholder in api_key / sensitive headers /
        # userinfo URLs. Without this the child would send the literal
        # "<redacted>" upstream and report an opaque 401 instead of the real
        # cause -- the injection env vars were never supplied.
        apply_endpoint_credentials(run, credentials, require_resolved=True)
        secrets.extend(credential_values(endpoint=run.cfg.endpoint))
        _run_single_benchmark(run)

    except KeyError as e:
        message = redact_credential_text(str(e), secrets)
        print(f"Error: Missing required config key: {message}", file=sys.stderr)
        sys.exit(1)
    except orjson.JSONDecodeError as e:
        message = redact_credential_text(str(e), secrets)
        print(f"Error: Invalid JSON in config file: {message}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:  # subprocess entry point: final safety net so the parent orchestrator gets a nonzero exit + traceback rather than an opaque crash
        import traceback

        print(
            f"Error: Failed to run benchmark: {redact_credential_text(str(e), secrets)}",
            file=sys.stderr,
        )
        # format_exc() rather than print_exc(): the traceback must pass through
        # the redactor before it reaches stderr, since locals repr'd in a frame
        # summary can carry the same credential values.
        print(
            redact_credential_text(traceback.format_exc(), secrets),
            file=sys.stderr,
            end="",
        )
        sys.exit(1)


def _script_entrypoint() -> None:
    """Real ``python -m aiperf.orchestrator.subprocess_runner`` entrypoint.

    Applies process-wide startup fixups that must happen before anything
    else in this process: the Windows event-loop policy switch must run
    before the first ``asyncio.run()`` this process makes (deep inside
    ``main()`` -> ``_run_single_benchmark`` -> ``bootstrap_and_run_service``),
    and pipe release must happen before ``main()`` does any real work.

    Split out from the ``if __name__ == "__main__":`` guard so it's directly
    unit-testable (module-level guards never execute under import/pytest).
    """
    from aiperf.common.event_loop import configure_event_loop_policy_for_platform

    # Must run before any asyncio.run()/uvloop.run() call in this process --
    # see aiperf.common.event_loop for the full rationale.
    configure_event_loop_policy_for_platform()

    # Release inherited pipe handles only when actually run as a subprocess
    # (`python -m aiperf.orchestrator.subprocess_runner ...`). Calling
    # ``main()`` from unit tests must NOT redirect stderr — pytest's capsys
    # needs to see the error prints, and there are no inherited pipes to
    # release in an in-process call. ``_release_inherited_pipes_on_windows``
    # itself is also gated on IS_WINDOWS so this is belt-and-suspenders.
    _release_inherited_pipes_on_windows()
    main()


if __name__ == "__main__":
    _script_entrypoint()
