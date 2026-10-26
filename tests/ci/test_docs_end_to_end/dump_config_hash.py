# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Print a stable hash of the parser-extracted server config.

Used by CI to skip the GPU test when a PR doesn't change anything the
parser would extract from the markdown tutorials.

Exits non-zero when the parser rejected anything, which the workflow reads
as "could not compute the hash" and runs the GPU job. A tree whose tags do
not parse has no meaningful fingerprint to compare, and the edit that broke
them is exactly the edit that must not skip.
"""

import hashlib
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent.parent
sys.path.insert(0, str(THIS_DIR))

from parser import MarkdownParser  # noqa: E402

# CI invokes this with no arguments. The optional root exists so the tests
# can fingerprint a purpose-built tree instead of the repository's own docs.
root = sys.argv[1] if len(sys.argv) > 1 else str(REPO_ROOT)
servers = MarkdownParser().parse_directory(root)

# Everything the harness would act on, not just the commands it would run.
# A fixture decides what a ``--config foo.yaml`` command benchmarks and a
# ``timeout=`` decides whether it is killed mid-run, so an edit to either
# changes the test without changing a command string. Fixture content is
# hashed rather than embedded to keep the digest input small; the path is
# kept in the clear because a renamed fixture is a different test.
normalized = sorted(
    [
        {
            "name": name,
            "setup": s.setup_command.command if s.setup_command else None,
            "health_check": (
                s.health_check_command.command if s.health_check_command else None
            ),
            # Include `weight` in each aiperf_command entry so weight-only
            # changes (which alter shard assignment and per-shard wall-clock)
            # invalidate the detect-changes hash and trigger the GPU e2e job.
            # Sorting as tuples keeps the result deterministic across runs.
            "aiperf_commands": sorted(
                [c.command, c.weight, c.timeout] for c in s.aiperf_commands
            ),
            "files": sorted(
                [f.path, hashlib.sha256(f.content.encode()).hexdigest()]
                for f in s.files
            ),
        }
        for name, s in servers.items()
    ],
    key=lambda d: d["name"],
)

rejected = sorted(
    name
    for name, s in servers.items()
    if s.parse_errors or s.fixture_conflicts or s.command_conflicts
)
if rejected:
    print(
        f"Parser rejected tags for: {', '.join(rejected)}; no comparable hash.",
        file=sys.stderr,
    )
    sys.exit(1)

print(hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest())
