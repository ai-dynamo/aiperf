# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A health check must fail the server, not just print that it failed.

Health-check blocks run under ``shell=True`` without ``-e``, so a block of
several steps exits with the status of its *last* command. A step gated with
``&& echo "reachable"`` therefore reports nothing when it times out: the
following step succeeds, the block exits 0, and a server that never came up
reads as healthy. Every wait must gate with ``|| { ...; exit 1; }`` instead.
"""

from __future__ import annotations

import re

from tests.unit.ci.docs_e2e_scan import REPO, markdown_files

_OPEN = re.compile(r"<!--\s*health-check-(\S+?)-endpoint-server")
_CLOSE = re.compile(r"<!--\s*/health-check-")


def _health_blocks() -> list[tuple[str, str, list[str]]]:
    """Every health-check block as ``(file, group, lines)``."""
    blocks = []
    for path in markdown_files():
        current: list[str] | None = None
        group = ""
        for line in path.read_text(encoding="utf-8").splitlines():
            if _CLOSE.search(line):
                if current is not None:
                    blocks.append((str(path.relative_to(REPO)), group, current))
                current = None
                continue
            if (match := _OPEN.search(line)) and current is None:
                current, group = [], match.group(1)
                continue
            if current is not None:
                current.append(line)
    return blocks


def test_no_health_check_step_is_gated_with_and_echo() -> None:
    offenders = [
        f"{file}: health-check-{group}: {line.strip()}"
        for file, group, body in _health_blocks()
        for line in body
        if re.search(r"&&\s*echo\b", line)
    ]
    assert not offenders, (
        "health-check steps gated with '&& echo' report success when the step "
        "fails, because the block exits with its last command's status. Use "
        "'|| { echo \"...\"; exit 1; }' instead:\n" + "\n".join(offenders)
    )
