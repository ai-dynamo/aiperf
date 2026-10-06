# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DO NOT MERGE: join-ordering instrumentation for the weka SPAWN_JOIN flake investigation."""

import os
import time
import traceback

import orjson

_PATH = os.environ.get("AIPERF_DBG_JOIN_LOG")
_FD: int | None = None


def dbg(event: str, **fields: object) -> None:
    global _FD
    if not _PATH:
        return
    fields["ev"] = event
    fields["t_ns"] = time.time_ns()
    fields["m_ns"] = time.perf_counter_ns()
    fields["pid"] = os.getpid()
    if _FD is None:
        _FD = os.open(_PATH, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    os.write(_FD, orjson.dumps(fields, default=str) + b"\n")


def caller(depth: int = 4) -> str:
    frames = traceback.extract_stack()[-(depth + 2) : -2]
    return " <- ".join(
        f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}" for f in reversed(frames)
    )
