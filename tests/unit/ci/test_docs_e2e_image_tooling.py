# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A tagged block may only invoke tools the test image actually installs.

``docs/reference/api-endpoints.md`` pipes a ``curl`` response into ``jq``.
The image installed ``curl`` and nothing else, so that line exited 127 on
every run. Nobody noticed, because the block ends in ``kill $AIPERF_PID``
and bash reported only the last command's status until ``set -e`` landed.

The watchlist below is explicit rather than derived: parsing every command
word out of blocks that embed heredocs, Python and shell loops produces more
noise than signal. A tool outside the list therefore slips through, which is
the cost of keeping this check readable. Add to the list when a guide starts
using a tool that is not in bash or coreutils.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

# Tools that are neither bash builtins nor part of a Debian base image, and
# so must be installed explicitly for a documented command to work.
WATCHLIST = ("jq", "curl", "git", "docker", "kubectl", "helm", "wget")


def _installed_in_test_stage() -> set[str]:
    """The packages the Dockerfile's ``test`` stage adds via apt-get."""
    dockerfile = (REPO_ROOT / "Dockerfile").read_text()
    stage = dockerfile.split("FROM env-builder AS test", 1)
    assert len(stage) == 2, "the Dockerfile no longer has a stage named 'test'"
    body = re.split(r"^FROM ", stage[1], flags=re.MULTILINE)[0]

    installed: set[str] = set()
    for match in re.finditer(r"apt-get install[^\n]*(?:\\\n[^\n]*)*", body):
        line = match.group(0).replace("\\\n", " ")
        for word in line.split():
            if word.startswith("-") or word in {"apt-get", "install", "&&"}:
                continue
            installed.add(word)
    return installed


@pytest.mark.parametrize("tool", WATCHLIST)  # fmt: skip
def test_a_tool_a_guide_invokes_is_in_the_test_image(tool: str) -> None:
    from parser import MarkdownParser

    parser = MarkdownParser()
    parser.parse_directory(str(REPO_ROOT))

    users = [
        command.file_path
        for server in parser.servers.values()
        for command in server.aiperf_commands
        if re.search(rf"(?:^|[|&;(\s]){re.escape(tool)}\s", command.command)
    ]
    if not users:
        pytest.skip(f"no tagged block invokes {tool}")

    assert tool in _installed_in_test_stage(), (
        f"{tool} is invoked by {sorted(set(users))} but the Dockerfile test "
        f"stage does not install it; the command will exit 127 in CI"
    )
