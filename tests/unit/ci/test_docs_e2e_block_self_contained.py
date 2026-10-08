# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A tagged block must not depend on a file another tagged block created.

Shards are packed per command, so two blocks of the same guide can run on
different machines. A block that writes ``trace.jsonl`` and a separate block
that reads it therefore fail whenever the packing separates them -- and
whether it does depends on how many commands exist elsewhere in the repo,
which makes it look like a flake rather than a missing file.

The convention is that a block creates the file it uses, in the same block.
Guides needing a file they cannot inline declare it with ``setup-file-``,
which is materialized into every shard.
"""

from __future__ import annotations

import re
import sys

from tests.unit.ci.docs_e2e_scan import REPO, markdown_files

sys.path.insert(0, str(REPO / "tests/ci/test_docs_end_to_end"))

from parser import MarkdownParser  # noqa: E402

_CREATES = re.compile(r"(?:cat\s*>|cat\s*<<\S*\s*>)\s*(\S+)")
_READS = re.compile(r"--input-file\s+(\S+)")


def test_no_block_reads_a_file_another_block_wrote() -> None:
    offenders: list[str] = []
    for doc in markdown_files():
        parser = MarkdownParser()
        parser._parse_file(str(doc))
        for server in parser.servers.values():
            commands = server.aiperf_commands
            written = {f for c in commands for f in _CREATES.findall(c.command)}
            fixtures = {f.path for f in server.files}
            for command in commands:
                borrowed = {
                    path
                    for path in _READS.findall(command.command)
                    if path in written
                    and path not in fixtures
                    and not re.search(r">\s*" + re.escape(path), command.command)
                }
                if borrowed:
                    rel = doc.relative_to(REPO)
                    offenders.append(
                        f"{rel}:{command.start_line} reads {sorted(borrowed)}, "
                        f"created by a different tagged block"
                    )

    assert not offenders, (
        "tagged blocks that depend on another block having run first; inline "
        "the file into the same block, or declare it with setup-file-:\n  "
        + "\n  ".join(offenders)
    )
