# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Low-overhead process entry point for AIPerf.

Offline/online Dynamo replay is authored through ``aiperf profile`` via
``transport.type: dynosim_offline`` / ``dynosim_online``; there is no
``aiperf dynosim`` fast path.
"""

from __future__ import annotations


def main(arguments: list[str] | None = None) -> int | None:
    """Dispatch one AIPerf invocation through the Cyclopts command tree."""
    from aiperf.cli import app

    if arguments is None:
        return app()
    return app(list(arguments))
