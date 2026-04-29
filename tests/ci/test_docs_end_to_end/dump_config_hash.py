# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Print a stable hash of the parser-extracted server config.

Used by CI to skip the GPU test when a PR doesn't change anything the
parser would extract from the markdown tutorials.
"""

import hashlib
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent.parent
sys.path.insert(0, str(THIS_DIR))

from parser import MarkdownParser  # noqa: E402

servers = MarkdownParser().parse_directory(str(REPO_ROOT))
normalized = sorted(
    [
        {
            "name": name,
            "setup": s.setup_command.command if s.setup_command else None,
            "health_check": (
                s.health_check_command.command if s.health_check_command else None
            ),
            "aiperf_commands": sorted(c.command for c in s.aiperf_commands),
        }
        for name, s in servers.items()
    ],
    key=lambda d: d["name"],
)
print(hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest())
