# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`aiperf` console-script launcher: hand off to the native Rust front door.

The wheel interns the native `aiperf-cli` binary at ``aiperf/_bin/aiperf-native``
(beside ``aiperf/_bin/aiperf-runner``). This shim resolves that binary via
``importlib.resources`` and replaces the current process with it (``os.execv`` on
POSIX; a spawn-and-wait on Windows), so ``aiperf profile`` / ``aiperf config`` run
entirely in Rust. The native front door itself delegates every other subcommand
back to ``python -m aiperf`` — which stays pure Python (``aiperf.entrypoint`` ->
``aiperf.cli:app``) and does NOT re-enter this launcher, so there is no exec loop.

If the interned binary is absent (e.g. a source checkout that never ran
``make bundle-cli``), fall back to the Python app so ``aiperf`` still works.
``AIPERF_NATIVE=0`` forces the Python app unconditionally (debugging / A-B).
"""

from __future__ import annotations

import os
import sys


def _native_binary() -> str | None:
    """Return the interned native binary path, or None when unavailable.

    Resolved through ``importlib.resources`` so it works from an installed wheel
    (where ``aiperf/_bin`` is package data) and from an editable install alike.
    """
    name = "aiperf-native.exe" if os.name == "nt" else "aiperf-native"
    try:
        from importlib.resources import as_file, files

        resource = files("aiperf").joinpath("_bin", name)
        with as_file(resource) as path:
            if path.is_file():
                return str(path)
    except (ModuleNotFoundError, FileNotFoundError, OSError):
        pass
    return None


def main(arguments: list[str] | None = None) -> int | None:
    """Exec the native front door, or fall back to the Python app."""
    argv = list(arguments) if arguments is not None else sys.argv[1:]

    if os.environ.get("AIPERF_NATIVE") != "0":
        binary = _native_binary()
        if binary is not None:
            if os.name == "nt":
                import subprocess

                return subprocess.run([binary, *argv]).returncode
            # POSIX: replace this process so the native binary owns the tty,
            # signal handling, and exit status directly.
            os.execv(binary, [binary, *argv])

    # No interned binary (or AIPERF_NATIVE=0): run the pure-Python app.
    from aiperf.entrypoint import main as python_main

    return python_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
