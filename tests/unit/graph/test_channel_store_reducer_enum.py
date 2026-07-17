# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R6 -- ``VersionedChannelStore`` uses ``ReducerName`` members, never ``.value``.

Repo contract: string enums are compared/looked-up with the member directly.
The functional tests pin that the enum-member registry lookup and the
overwrite-conflict compare keep working; the mechanical test rejects a
re-introduction of ``.value`` access on the reducer field.
"""

from __future__ import annotations

import inspect

import pytest

from aiperf.dataset.graph.models import ChannelSpec, ReducerName
from aiperf.graph import channel_store as channel_store_module
from aiperf.graph.channel_store import VersionedChannelStore
from aiperf.graph.reducers import OverwriteConflictError


def _store(specs: dict[str, ChannelSpec], producers: dict[str, int]):
    return VersionedChannelStore(
        initial={},
        channel_specs=specs,
        producers_per_channel=producers,
    )


def test_no_reducer_value_access_in_channel_store_source():
    """String-enum contract: no ``.value`` on the ``ReducerName`` field."""
    src = inspect.getsource(channel_store_module)
    assert "reducer.value" not in src


def test_add_messages_reducer_resolved_from_enum_member():
    """The registry lookup accepts the ``ReducerName`` member directly."""
    store = _store({"msgs": ChannelSpec(reducer=ReducerName.ADD_MESSAGES)}, {"msgs": 2})
    store.write(["msgs"], [{"role": "user", "content": "a"}], writer_node_id="n1")
    store.write(["msgs"], [{"role": "user", "content": "b"}], writer_node_id="n2")

    snapshot = store.snapshot()

    assert [m["content"] for m in snapshot["msgs"]] == ["a", "b"]


def test_overwrite_conflict_detected_via_enum_member_compare():
    """The overwrite second-writer rejection keys on the enum member."""
    store = _store({"out": ChannelSpec(reducer=ReducerName.OVERWRITE)}, {"out": 2})
    store.write(["out"], "first", writer_node_id="n1")

    with pytest.raises(OverwriteConflictError):
        store.write(["out"], "second", writer_node_id="n2")

    assert store.snapshot()["out"] == "first"
