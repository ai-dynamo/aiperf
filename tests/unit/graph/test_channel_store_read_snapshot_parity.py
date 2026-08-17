# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``read`` at a full capture must agree with ``snapshot`` by construction.

Both go through ``VersionedChannelStore._reduce``: same init seed, same
``(write_seq, writer_node_id)`` order, same no-writes shortcut. These tests
would fail if the two read paths ever diverged on seeding or ordering.
"""

from __future__ import annotations

import pytest

from aiperf.dataset.graph.models import ChannelRequirement, ChannelSpec, ReducerName
from aiperf.graph.channel_store import VersionedChannelStore
from aiperf.graph.reducers import UNSET


def _store(
    specs: dict[str, ChannelSpec],
    producers: dict[str, int],
    initial: dict[str, object] | None = None,
) -> VersionedChannelStore:
    """A store over the given specs, producer counts, and optional init seed."""
    return VersionedChannelStore(
        initial=dict(initial or {}),
        channel_specs=specs,
        producers_per_channel=producers,
    )


async def _read_all(store: VersionedChannelStore, channel: str) -> object:
    """Capture every declared producer write on `channel` and reduce it."""
    reqs = [ChannelRequirement(channel=channel, count="all")]
    capture = await store.await_inputs(reqs)
    return store.read(reqs, capture)[channel]


@pytest.mark.asyncio
async def test_read_full_capture_matches_snapshot_with_init_seed() -> None:
    """The init seed feeds both paths identically as the reducer's start value."""
    store = _store(
        {"msgs": ChannelSpec(reducer=ReducerName.ADD_MESSAGES)},
        {"msgs": 2},
        initial={"msgs": [{"role": "system", "content": "seed"}]},
    )
    store.write(["msgs"], [{"role": "user", "content": "a"}], writer_node_id="n1")
    store.write(["msgs"], [{"role": "user", "content": "b"}], writer_node_id="n2")

    read_value = await _read_all(store, "msgs")

    assert [m["content"] for m in read_value] == ["seed", "a", "b"]
    assert store.snapshot()["msgs"] == read_value


@pytest.mark.asyncio
async def test_read_full_capture_matches_snapshot_write_order() -> None:
    """Both paths order writes by ``(write_seq, writer_node_id)``."""
    store = _store({"msgs": ChannelSpec(reducer=ReducerName.ADD_MESSAGES)}, {"msgs": 3})
    # Writer ids intentionally out of lexical order: commit seq must win.
    store.write(["msgs"], [{"role": "user", "content": "z"}], writer_node_id="zz")
    store.write(["msgs"], [{"role": "user", "content": "a"}], writer_node_id="aa")
    store.write(["msgs"], [{"role": "user", "content": "m"}], writer_node_id="mm")

    read_value = await _read_all(store, "msgs")

    assert [m["content"] for m in read_value] == ["z", "a", "m"]
    assert store.snapshot()["msgs"] == read_value


@pytest.mark.asyncio
async def test_no_writes_no_seed_is_unset_on_both_paths() -> None:
    """An unwritten, unseeded channel reduces to ``UNSET``, not a reducer call."""
    store = _store({"out": ChannelSpec(reducer=ReducerName.OVERWRITE)}, {"out": 0})

    read_value = await _read_all(store, "out")

    assert read_value is UNSET
    assert store.snapshot()["out"] is UNSET


@pytest.mark.asyncio
async def test_partial_capture_is_a_prefix_of_the_snapshot() -> None:
    """A capture below the write count reduces only its own subset."""
    store = _store({"msgs": ChannelSpec(reducer=ReducerName.ADD_MESSAGES)}, {"msgs": 2})
    store.write(["msgs"], [{"role": "user", "content": "a"}], writer_node_id="n1")

    reqs = [ChannelRequirement(channel="msgs", count=1)]
    capture = await store.await_inputs(reqs)
    store.write(["msgs"], [{"role": "user", "content": "b"}], writer_node_id="n2")

    assert [m["content"] for m in store.read(reqs, capture)["msgs"]] == ["a"]
    assert [m["content"] for m in store.snapshot()["msgs"]] == ["a", "b"]
