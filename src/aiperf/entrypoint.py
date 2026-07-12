# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Low-overhead process entry point for AIPerf.

DynoSim owns its replay and live-mocker parsers. Their raw forwarding paths run
before the general Cyclopts command tree is imported so the AIPerf namespace
does not add unrelated startup CPU or memory to canonical Dynamo processes.
"""

from __future__ import annotations

import os
import sys
from typing import NoReturn

_DYNOSIM_MODULES = {
    "mocker": "dynamo.mocker",
    "run": "dynamo.replay",
}


def _import_dynamo_symbol(module_name: str, symbol: str):
    try:
        module = __import__(module_name, fromlist=[symbol])
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "DynoSim support requires an ai-dynamo installation containing "
            f"{module_name!r}; install Dynamo in the AIPerf environment"
        ) from error
    return getattr(module, symbol)


def _run_dynosim(arguments: list[str]) -> int | None:
    operation, *forwarded = arguments
    if operation == "run":
        status = _import_dynamo_symbol("dynamo.replay.main", "main")(forwarded)
    elif operation == "mocker":
        main = _import_dynamo_symbol("dynamo.mocker.main", "main")
        previous = sys.argv
        sys.argv = ["aiperf dynosim mocker", *forwarded]
        try:
            status = main()
        finally:
            sys.argv = previous
    else:
        raise AssertionError(f"unsupported fast DynoSim operation {operation!r}")
    if status not in (None, 0):
        raise SystemExit(status)
    return status


def _exec_dynosim(arguments: list[str]) -> NoReturn:
    """Replace the CLI shim with the corresponding canonical Dynamo process."""
    operation, *forwarded = arguments
    module = _DYNOSIM_MODULES[operation]
    os.execv(
        sys.executable,
        [sys.executable, "-m", module, *forwarded],
    )
    raise AssertionError("os.execv returned")


def main(arguments: list[str] | None = None) -> int | None:
    """Dispatch one AIPerf invocation, fast-pathing raw Dynamo products."""
    authored = list(sys.argv[1:] if arguments is None else arguments)
    if (
        len(authored) >= 2
        and authored[0] == "dynosim"
        and authored[1] in _DYNOSIM_MODULES
    ):
        if arguments is None:
            _exec_dynosim(authored[1:])
        return _run_dynosim(authored[1:])

    from aiperf.cli import app

    if arguments is None:
        return app()
    return app(authored)
