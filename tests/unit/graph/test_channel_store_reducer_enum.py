# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R6 -- ``VersionedChannelStore`` uses ``ReducerName`` members, never ``.value``."""

from __future__ import annotations

import inspect

import pytest

from aiperf.dataset.graph.models import ChannelSpec, ReducerName
from aiperf.graph import channel_store as channel_store_module
from aiperf.graph.channel_store import VersionedChannelStore
from aiperf.graph.reducers import OverwriteConflictError


def _store(
    specs: dict[str, ChannelSpec], producers: dict[str, int]
) -> VersionedChannelStore:
    """A store over the given channel specs and per-channel producer counts."""
    return VersionedChannelStore(
        initial={},
        channel_specs=specs,
        producers_per_channel=producers,
    )


def test_no_reducer_value_access_in_channel_store_source() -> None:
    """Source-text guard: the channel_store module never spells ``reducer.value``."""
    src = inspect.getsource(channel_store_module)
    assert "reducer.value" not in src


def test_add_messages_reducer_resolved_from_enum_member() -> None:
    """The registry lookup accepts the ``ReducerName`` member directly."""
    store = _store({"msgs": ChannelSpec(reducer=ReducerName.ADD_MESSAGES)}, {"msgs": 2})
    store.write(["msgs"], [{"role": "user", "content": "a"}], writer_node_id="n1")
    store.write(["msgs"], [{"role": "user", "content": "b"}], writer_node_id="n2")

    snapshot = store.snapshot()

    assert [m["content"] for m in snapshot["msgs"]] == ["a", "b"]


def test_overwrite_conflict_detected_via_enum_member_compare() -> None:
    """The overwrite second-writer rejection keys on the enum member."""
    store = _store({"out": ChannelSpec(reducer=ReducerName.OVERWRITE)}, {"out": 2})
    store.write(["out"], "first", writer_node_id="n1")

    with pytest.raises(OverwriteConflictError):
        store.write(["out"], "second", writer_node_id="n2")

    assert store.snapshot()["out"] == "first"


def test_overwrite_conflict_names_two_distinct_writers() -> None:
    """The rejection message must identify the incumbent and the interloper."""
    store = _store({"out": ChannelSpec(reducer=ReducerName.OVERWRITE)}, {"out": 2})
    store.write(["out"], "first", writer_node_id="n1")

    with pytest.raises(OverwriteConflictError) as exc:
        store.write(["out"], "second", writer_node_id="n2")

    message = str(exc.value)
    assert "'n1'" in message and "'n2'" in message


def test_same_node_rewrite_reports_an_invariant_violation_not_a_second_writer() -> None:
    """A repeat write from the SAME node is still an error -- with its own message.

    Each node fires at most once per instance run, so this is an executor
    invariant violation. It shares the exception type with a genuine
    multi-writer conflict but must not share its wording: the old message named
    one node as both incumbent and interloper ("already written by 'n1';
    rejecting second writer 'n1'"), sending the reader hunting for a second
    writer that does not exist.
    """
    store = _store({"out": ChannelSpec(reducer=ReducerName.OVERWRITE)}, {"out": 2})
    store.write(["out"], "first", writer_node_id="n1")

    with pytest.raises(OverwriteConflictError) as exc:
        store.write(["out"], "second", writer_node_id="n1")

    message = str(exc.value)
    assert "same node" in message
    assert "invariant violation" in message
    assert "rejecting second writer" not in message
