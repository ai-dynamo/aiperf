# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply namespace-local resource limits before importing AIPerf worker code.

The stock runner executes this file by absolute path after Bubblewrap has
created the evaluator's private user and PID namespaces.  Keeping the bootstrap
standalone avoids importing the :mod:`aiperf` package before ``RLIMIT_NPROC``
is installed.  The remaining fixed arguments belong to the evaluator worker.
"""

from __future__ import annotations

import resource
import sys
from collections.abc import Sequence

_MAX_PROCESSES_FLAG = "--max-processes"


def _parse_max_processes(value: str) -> int:
    """Parse one canonical positive decimal process ceiling."""
    if not value.isascii() or not value.isdecimal() or value.startswith("0"):
        raise ValueError("--max-processes must be a canonical positive integer")
    maximum = int(value)
    if maximum <= 0:
        raise ValueError("--max-processes must be a canonical positive integer")
    return maximum


def _install_process_limit(maximum: int) -> None:
    """Install and read back one exact soft/hard namespace process ceiling."""
    _, inherited_hard = resource.getrlimit(resource.RLIMIT_NPROC)
    if inherited_hard != resource.RLIM_INFINITY and inherited_hard < maximum:
        raise RuntimeError("inherited RLIMIT_NPROC cannot satisfy the stock ceiling")
    expected = (maximum, maximum)
    resource.setrlimit(resource.RLIMIT_NPROC, expected)
    if resource.getrlimit(resource.RLIMIT_NPROC) != expected:
        raise RuntimeError("RLIMIT_NPROC did not retain the exact stock ceiling")


def main(argv: Sequence[str] | None = None) -> None:
    """Install the leading resource policy, then enter the worker bootstrap."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) < 3 or arguments[0] != _MAX_PROCESSES_FLAG:
        raise ValueError("resource bootstrap requires leading --max-processes VALUE")
    maximum = _parse_max_processes(arguments[1])
    _install_process_limit(maximum)

    from aiperf.accuracy.evaluation.worker import main as worker_main

    worker_main(arguments[2:])


if __name__ == "__main__":
    main()
