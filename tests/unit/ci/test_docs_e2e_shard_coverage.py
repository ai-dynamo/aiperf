# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every tagged server must have a matrix shard, and vice versa.

A server defined in the docs but missing from the workflow matrix is the worst
kind of gap: its guides look covered, the tag guard is satisfied, and nothing
ever executes them. A shard naming a server that no longer exists is the
mirror failure -- the job boots a runner and finds nothing to do.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
HARNESS = REPO / "tests/ci/test_docs_end_to_end"
sys.path.insert(0, str(HARNESS))

from parser import MarkdownParser  # noqa: E402

# Long-running guides live in their own weekly workflow so they cannot be
# pulled in by nightly's workflow_call; both files must be scanned or their
# server reads as unsharded.
WORKFLOWS = (
    REPO / ".github/workflows/test-docs-end-to-end.yml",
    REPO / ".github/workflows/test-docs-long-guides.yml",
)


def _documented_servers() -> set[str]:
    parser = MarkdownParser()
    parser.parse_directory(str(REPO))
    # Only servers that can actually boot are runnable; a bare file-fixture or
    # run tag without a setup block is a doc bug caught by its own test.
    return {
        name
        for name, server in parser.servers.items()
        if server.setup_command is not None
    }


def _matrix_servers() -> set[str]:
    """Every server named by a matrix shard in any docs-e2e workflow."""
    servers: set[str] = set()
    for path in WORKFLOWS:
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        for name, job in spec["jobs"].items():
            if not name.startswith("test-docs-end-to-end"):
                continue
            shards = job.get("strategy", {}).get("matrix", {}).get("shard")
            if shards:
                servers.update(shard["server"] for shard in shards)
    return servers


def test_every_documented_server_has_a_shard() -> None:
    missing = _documented_servers() - _matrix_servers()
    assert not missing, (
        f"servers defined in docs but absent from the workflow matrix: "
        f"{sorted(missing)}. Their tagged commands would never run."
    )


def test_every_shard_points_at_a_real_server() -> None:
    dangling = _matrix_servers() - _documented_servers()
    assert not dangling, (
        f"matrix shards with no setup block in docs: {sorted(dangling)}"
    )
