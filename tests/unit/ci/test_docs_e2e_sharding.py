# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The docs-e2e sharder must never split a document across shards.

Guides routinely split setup and use across two tagged blocks --
``trace-replay.md`` writes ``custom_trace.jsonl`` in one block and consumes it
in the next. Packing individual commands can place the consumer in a shard that
never ran the producer, so the guide fails on a missing file, and whether it
happens depends on how many commands exist in total elsewhere in the repo.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _Cmd:
    file_path: str
    start_line: int
    weight: int = 80


def _shard(commands: list[_Cmd], shard_total: int) -> list[list[_Cmd]]:
    """Mirror of the packing in main.py, exercised directly."""
    bins: list[list[_Cmd]] = [[] for _ in range(shard_total)]
    load = [0] * shard_total
    docs: dict[str, list[_Cmd]] = {}
    for cmd in commands:
        docs.setdefault(cmd.file_path, []).append(cmd)
    for cmds in docs.values():
        cmds.sort(key=lambda c: c.start_line)
    for _, cmds in sorted(
        docs.items(), key=lambda kv: (-sum(c.weight for c in kv[1]), kv[0])
    ):
        target = min(range(shard_total), key=lambda i: load[i])
        bins[target].extend(cmds)
        load[target] += sum(c.weight for c in cmds)
    return bins


def _commands() -> list[_Cmd]:
    return [
        _Cmd("docs/trace-replay.md", 66),  # writes custom_trace.jsonl
        _Cmd("docs/trace-replay.md", 77),  # consumes it
        _Cmd("docs/trace-replay.md", 198),
        _Cmd("docs/a.md", 10, 300),
        _Cmd("docs/b.md", 10, 200),
        _Cmd("docs/c.md", 10, 100),
        _Cmd("docs/d.md", 10, 90),
    ]


def test_a_documents_commands_always_land_in_one_shard() -> None:
    for shard_total in (2, 3, 4, 5, 8):
        bins = _shard(_commands(), shard_total)
        placement: dict[str, int] = {}
        for index, shard in enumerate(bins):
            for cmd in shard:
                previous = placement.setdefault(cmd.file_path, index)
                assert previous == index, (
                    f"{cmd.file_path} split across shards {previous} and "
                    f"{index} at shard_total={shard_total}"
                )


def test_producer_precedes_consumer_within_a_shard() -> None:
    """Ordering inside a document must follow the page, not the weight."""
    for shard in _shard(_commands(), 4):
        lines = [c.start_line for c in shard if c.file_path == "docs/trace-replay.md"]
        assert lines == sorted(lines)


def test_every_command_is_placed_exactly_once() -> None:
    commands = _commands()
    for shard_total in (1, 2, 3, 4, 8):
        placed = [c for shard in _shard(commands, shard_total) for c in shard]
        assert len(placed) == len(commands)
        assert {(c.file_path, c.start_line) for c in placed} == {
            (c.file_path, c.start_line) for c in commands
        }


def test_packing_is_deterministic() -> None:
    """Filesystem iteration order must not change shard assignment."""
    forward = _shard(_commands(), 4)
    reverse = _shard(list(reversed(_commands())), 4)
    as_keys = lambda bins: [  # noqa: E731
        sorted((c.file_path, c.start_line) for c in shard) for shard in bins
    ]
    assert as_keys(forward) == as_keys(reverse)
