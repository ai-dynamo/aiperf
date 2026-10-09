# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A shard must never receive a command whose fixture it did not create.

Guides build their own inputs in one tagged block and consume them in the
next: ``docs/benchmark-modes/trace-replay.md`` writes ``custom_trace.jsonl``
with a heredoc, then profiles against it. Both blocks carry ``aiperf-run-``
tags, so the sharder sees two independent commands and is free to put them on
different runners -- which it did, failing shard 3/4 on 2026-10-09 the moment
container cleanup started removing the leftover container that had been
carrying the file between jobs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HARNESS = REPO_ROOT / "tests/ci/test_docs_end_to_end"


@pytest.fixture(autouse=True)
def _contain_import_side_effects():
    """The harness imports its modules flatly (``from parser import ...``),
    so its directory has to go on ``sys.path``; those names are generic enough
    to shadow unrelated modules for the rest of the xdist worker."""
    saved_path = list(sys.path)
    saved_modules = set(sys.modules)
    sys.path.insert(0, str(HARNESS))
    try:
        yield
    finally:
        sys.path[:] = saved_path
        for name in set(sys.modules) - saved_modules:
            del sys.modules[name]


def _real_commands(server: str = "vllm-default-openai") -> list:
    """Parse what production parses.

    ``main.py`` calls ``parse_directory(repo_root)``, not ``docs/``: tags are
    collected from every markdown file in the repo. Narrowing this to ``docs/``
    would shard a different command set than CI does, and the counts would
    silently diverge.
    """
    from parser import MarkdownParser

    parser = MarkdownParser()
    parser.parse_directory(str(REPO_ROOT))
    return parser.servers[server].aiperf_commands


@pytest.mark.parametrize("shard_total", [2, 3, 4, 8])  # fmt: skip
def test_a_guide_is_never_split_across_shards(shard_total: int) -> None:
    from main import shard_commands

    commands = _real_commands()
    owner: dict[str, int] = {}
    for index in range(shard_total):
        mine, _ = shard_commands(commands, index, shard_total)
        for cmd in mine:
            previous = owner.setdefault(cmd.file_path, index)
            assert previous == index, (
                f"{cmd.file_path} is split across shards {previous} and "
                f"{index}; a fixture built in one shard is missing in the "
                f"other"
            )


@pytest.mark.parametrize("shard_total", [2, 3, 4, 8])  # fmt: skip
def test_every_command_runs_exactly_once(shard_total: int) -> None:
    """Grouping must not drop or duplicate work."""
    from main import shard_commands

    commands = _real_commands()
    seen = [
        (c.file_path, c.start_line)
        for index in range(shard_total)
        for c in shard_commands(commands, index, shard_total)[0]
    ]
    expected = [(c.file_path, c.start_line) for c in commands]
    assert sorted(seen) == sorted(expected)
    assert len(seen) == len(set(seen)), "a command was assigned to two shards"


@pytest.mark.parametrize("shard_total", [2, 3, 4, 8])  # fmt: skip
def test_the_trace_replay_fixture_ships_with_its_consumer(shard_total: int) -> None:
    """The concrete pair that failed in CI.

    Parametrised because a single shard count proves little: under the old
    per-command packing this pair happened to stay together at some counts and
    split at others, so a test pinned to one would have passed on the bug.
    """
    from main import shard_commands

    commands = _real_commands()
    for index in range(shard_total):
        mine, _ = shard_commands(commands, index, shard_total)
        bodies = [c.command for c in mine]
        writes = any("custom_trace.jsonl << " in b for b in bodies)
        reads = any("--input-file custom_trace.jsonl" in b for b in bodies)
        assert reads <= writes, (
            f"shard {index + 1}/{shard_total} profiles against "
            f"custom_trace.jsonl but never creates it"
        )
        if writes:
            consumer = next(
                i for i, b in enumerate(bodies) if "--input-file custom_trace" in b
            )
            creator = next(
                i for i, b in enumerate(bodies) if "custom_trace.jsonl << " in b
            )
            assert creator < consumer, "the fixture must run before its consumer"


def test_shards_stay_balanced() -> None:
    """Grouping by file must not make one runner the wall-clock bottleneck."""
    from main import shard_commands

    _, load = shard_commands(_real_commands(), 0, 4)
    assert max(load) <= 1.35 * (sum(load) / len(load)), (
        f"shard weights are lopsided: {load}"
    )
