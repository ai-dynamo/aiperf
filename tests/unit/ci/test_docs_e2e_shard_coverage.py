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
                # A sweep shard names a model-sweep target rather than a
                # documented server: it replays another group's commands
                # against a different family, so its server is synthetic.
                servers.update(shard["server"] for shard in shards if "server" in shard)
    return servers


def _matrix_sweeps() -> set[str]:
    """Every model-sweep target named by a matrix shard."""
    sweeps: set[str] = set()
    for path in WORKFLOWS:
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        for name, job in spec["jobs"].items():
            if not name.startswith("test-docs-end-to-end"):
                continue
            shards = job.get("strategy", {}).get("matrix", {}).get("shard") or []
            sweeps.update(shard["sweep"] for shard in shards if "sweep" in shard)
    return sweeps


def test_every_sweep_shard_names_a_real_target() -> None:
    """A sweep shard pointing at nothing would run zero commands and pass."""
    import sys
    from pathlib import Path as _Path

    harness = _Path(__file__).resolve().parents[3] / "tests/ci/test_docs_end_to_end"
    sys.path.insert(0, str(harness))
    from model_sweep import SWEEP_TARGETS

    dangling = _matrix_sweeps() - set(SWEEP_TARGETS)
    assert not dangling, (
        f"matrix sweep shards with no target in model_sweep.SWEEP_TARGETS: "
        f"{sorted(dangling)}"
    )


def test_every_sweep_target_replays_a_documented_server() -> None:
    """The corpus it borrows must exist, or the sweep tests nothing."""
    import sys
    from pathlib import Path as _Path

    harness = _Path(__file__).resolve().parents[3] / "tests/ci/test_docs_end_to_end"
    sys.path.insert(0, str(harness))
    from model_sweep import SWEEP_TARGETS

    documented = _documented_servers()
    for name, target in SWEEP_TARGETS.items():
        assert target.replays in documented, (
            f"sweep '{name}' replays '{target.replays}', which is not a "
            f"documented server group"
        )


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
