# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph reducers for overwrite and add_messages channels."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Sentinel marking a channel that has never been written. Compare with
# ``is UNSET``; msgspec's singleton keeps identity across pickle/deepcopy.
from msgspec import UNSET

Reducer = Callable[[Any, list[tuple[str, Any]]], Any]


class OverwriteConflictError(ValueError):
    """Raised when two or more nodes write an overwrite-typed channel concurrently."""


class ReducerNameError(ValueError):
    """Raised when a reducer name is unknown or reserved for a future spec version."""


def overwrite_reducer(current: Any, writes: list[tuple[str, Any]]) -> Any:
    """Return the single writer's value; reject concurrent multi-writer."""
    if not writes:
        return current
    if len(writes) > 1:
        ids = ", ".join(w[0] for w in writes)
        raise OverwriteConflictError(
            f"overwrite-typed channel written by multiple nodes concurrently: {ids}"
        )
    return writes[0][1]


class MessageList(list):
    """`list` subclass for add_messages accumulators with an `id -> index` map.

    The map (`_id_index`) is internal bookkeeping that lets
    `add_messages_reducer` replace by-id in O(1) instead of O(n). Channel
    readers consume the value as a plain list — `isinstance(x, list)`
    passes and all list ops behave normally.

    The index is rebuilt on `__init__`/`__deepcopy__`/`__reduce__` so
    copy/pickle round-trips stay correct even when the resulting object
    is a `MessageList`. Mutating the list directly (append, __setitem__,
    etc.) bypasses the index, so all production mutation must go through
    the reducer; the reducer is the only writer.
    """

    __slots__ = ("_id_index",)

    def __init__(self, iterable: Any = ()) -> None:
        super().__init__(iterable)
        self._id_index: dict[Any, int] = {}
        for i, msg in enumerate(self):
            if isinstance(msg, dict):
                msg_id = msg.get("id")
                if msg_id is not None:
                    self._id_index[msg_id] = i

    def __deepcopy__(self, memo: dict[int, Any]) -> MessageList:
        import copy as _copy

        return MessageList(_copy.deepcopy(list(self), memo))

    def __reduce__(self) -> tuple[Any, tuple[list[Any]]]:
        return (MessageList, (list(self),))


def add_messages_reducer(
    current: Any, writes: list[tuple[str, Any]]
) -> list[dict[str, Any]]:
    """Append writer values; replace prior messages whose `id` matches a new message.

    Maintains an `id -> index` map on the accumulator (`MessageList`) so
    id-based replacement is O(1). Messages without `id` take the plain
    fast-append path with no dict overhead beyond a sentinel check.
    """
    if isinstance(current, MessageList):
        accumulator = current
    elif current is UNSET:
        accumulator = MessageList()
    else:
        accumulator = MessageList(current)
    id_index = accumulator._id_index
    for writer_id, value in writes:
        if not isinstance(value, list):
            raise TypeError(
                f"add_messages reducer expected a list for writer '{writer_id}', got {type(value).__name__}"
            )
        for msg in value:
            msg_id = msg.get("id") if isinstance(msg, dict) and "id" in msg else None
            if msg_id is not None:
                idx = id_index.get(msg_id)
                if idx is not None:
                    accumulator[idx] = msg
                    continue
                id_index[msg_id] = len(accumulator)
            accumulator.append(msg)
    return accumulator


_REDUCERS: dict[str, Reducer] = {
    "overwrite": overwrite_reducer,
    "add_messages": add_messages_reducer,
}


def get_reducer(name: str) -> Reducer:
    """Look up a reducer function by name; raise ReducerNameError if unknown."""
    if name not in _REDUCERS:
        raise ReducerNameError(f"unknown reducer '{name}'")
    return _REDUCERS[name]
