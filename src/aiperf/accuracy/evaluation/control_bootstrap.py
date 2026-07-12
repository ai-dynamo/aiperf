# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attested pre-import bootstrap for anonymous evaluator control pipes.

Bubblewrap preserves only standard descriptors across its own exec boundary.
Rust therefore binds the request pipe to stdin and the response pipe to stdout.
This bootstrap runs under ``python -I -S`` before site initialization, moves
those one-way pipes to the worker's fixed descriptors, reserves ordinary
stdin/stdout, closes every unrelated descriptor, and only then admits the
pinned site-packages tree and starts the evaluator worker.
"""

from __future__ import annotations

import fcntl
import os
import resource
import runpy
import stat
import sys

_CONTROL_READ_FD = 3
_CONTROL_WRITE_FD = 4
_PROCESS_LIMIT_ENV = "AIPERF_EVALUATOR_BOOTSTRAP_PROCESS_LIMIT"
_MAX_DESCRIPTOR_SCAN = 1_048_576
_PINNED_SITE_PACKAGES = "/runtime/lib/python3.12/site-packages"
_WORKER_MODULE = "aiperf.accuracy.evaluation.worker"


def _require_anonymous_pipe(fd: int, access: int, label: str) -> None:
    metadata = os.fstat(fd)
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    if not stat.S_ISFIFO(metadata.st_mode) or flags & os.O_ACCMODE != access:
        raise RuntimeError(f"{label} is not the expected anonymous one-way pipe")


def _descriptor_ceiling() -> int:
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard == resource.RLIM_INFINITY:
        return _MAX_DESCRIPTOR_SCAN
    return min(int(hard), _MAX_DESCRIPTOR_SCAN)


def _reserve_control_descriptors() -> None:
    _require_anonymous_pipe(sys.stdin.fileno(), os.O_RDONLY, "control request input")
    _require_anonymous_pipe(sys.stdout.fileno(), os.O_WRONLY, "control response output")
    os.dup2(sys.stdin.fileno(), _CONTROL_READ_FD, inheritable=False)
    os.dup2(sys.stdout.fileno(), _CONTROL_WRITE_FD, inheritable=False)

    null_fd = os.open("/dev/null", os.O_RDWR | os.O_CLOEXEC)
    try:
        os.dup2(null_fd, sys.stdin.fileno(), inheritable=False)
        os.dup2(null_fd, sys.stdout.fileno(), inheritable=False)
    finally:
        if null_fd > _CONTROL_WRITE_FD:
            os.close(null_fd)

    os.set_inheritable(_CONTROL_READ_FD, False)
    os.set_inheritable(_CONTROL_WRITE_FD, False)
    os.closerange(_CONTROL_WRITE_FD + 1, _descriptor_ceiling())


def _apply_process_limit() -> None:
    raw_limit = os.environ.pop(_PROCESS_LIMIT_ENV, None)
    if (
        raw_limit is None
        or not raw_limit.isascii()
        or not raw_limit.isdecimal()
        or raw_limit != str(int(raw_limit))
        or int(raw_limit) <= 0
    ):
        raise RuntimeError("evaluator bootstrap process limit was absent or invalid")
    requested = int(raw_limit)
    _, inherited_hard = resource.getrlimit(resource.RLIMIT_NPROC)
    applied = (
        requested
        if inherited_hard == resource.RLIM_INFINITY
        else min(requested, int(inherited_hard))
    )
    resource.setrlimit(resource.RLIMIT_NPROC, (applied, applied))
    if resource.getrlimit(resource.RLIMIT_NPROC) != (applied, applied):
        raise RuntimeError("evaluator bootstrap process hard limit was not enforced")


def main() -> None:
    if not (
        sys.flags.isolated == 1
        and sys.flags.no_site == 1
        and sys.flags.safe_path
    ):
        raise RuntimeError(
            "evaluator control bootstrap requires python -I -S safe-path isolation"
        )
    _reserve_control_descriptors()
    _apply_process_limit()
    if sys.path.count(_PINNED_SITE_PACKAGES) != 0:
        raise RuntimeError("pinned site-packages was visible before bootstrap admission")
    sys.path.insert(0, _PINNED_SITE_PACKAGES)
    sys.argv[0] = _WORKER_MODULE
    runpy.run_module(_WORKER_MODULE, run_name="__main__", alter_sys=False)


if __name__ == "__main__":
    main()
