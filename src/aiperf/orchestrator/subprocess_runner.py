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
    injected_headers_raw = os.environ.pop(_INJECTED_HEADERS_ENV, None)
    injected_headers = (
        orjson.loads(injected_headers_raw) if injected_headers_raw else None
    )

    from aiperf.cli_runner import _run_single_benchmark
    from aiperf.config import BenchmarkRun

    try:
        with open(config_file, "rb") as f:
            data = orjson.loads(f.read())

        run = BenchmarkRun.model_validate(data)
        if injected_api_key is not None:
            run.cfg.endpoint.api_key = injected_api_key
        if injected_headers:
            run.cfg.endpoint.headers.update(injected_headers)
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
    main()
