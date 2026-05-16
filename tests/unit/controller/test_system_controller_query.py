# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SystemController GET_POD_STATES @on_command handler.

Locks in:
- the handler subscribes to CommandType.GET_POD_STATES (so a future enum
  rename or decorator drift fails loudly).
- the returned dict shape is what /api/progress and /api/debug/* expect:
  ``{"pod_states": {pod_index: <model_dump>}, "worker_startup_states": {...}}``.
"""

from __future__ import annotations

from types import SimpleNamespace

import orjson
import pytest

from aiperf.common.control_structs import Command
from aiperf.common.enums import CommandType
from aiperf.common.messages import WorkerPodStateMessage
from aiperf.controller.system_controller_query import SystemControllerQueryMixin


def _pod(pod_index: str, *, declared: int, ready: int) -> WorkerPodStateMessage:
    return WorkerPodStateMessage(
        service_id=f"wpm-{pod_index}",
        pod_index=pod_index,
        benchmark_generation="g",
        dataset_generation="d",
        declared_workers=declared,
        declared_record_processors=1,
        router_connected_workers=ready,
        dispatchable_workers=ready,
        ready_workers=ready,
        ready_record_processors=1,
        degraded_workers=max(0, declared - ready),
        degraded_record_processors=0,
        pod_state="ready" if ready >= 1 else "starting",
        admission_state="dispatchable" if ready >= 1 else "admitting",
    )


def _command() -> Command:
    return Command(cid="cid-1", cmd=CommandType.GET_POD_STATES)


@pytest.mark.asyncio
async def test_handler_subscribed_to_get_pod_states() -> None:
    """The decorator must record CommandType.GET_POD_STATES — without it,
    inbound RPCs from the API service silently get a CommandAck instead
    of running the handler."""
    handler = SystemControllerQueryMixin._on_get_pod_states
    params = getattr(handler, "__aiperf_hook_params__", ())
    assert CommandType.GET_POD_STATES in params


@pytest.mark.asyncio
async def test_returns_pod_states_and_startup_states() -> None:
    """Handler returns the dict shape /api/progress and /api/debug expect."""
    instance = SimpleNamespace(
        _pod_states={
            "0": _pod("0", declared=4, ready=4),
            "1": _pod("1", declared=4, ready=2),
        },
        _worker_startup_states={"w-0": "ready", "w-1": "waiting_for_dataset"},
    )

    result = await SystemControllerQueryMixin._on_get_pod_states(instance, _command())

    assert set(result.keys()) == {"pod_states", "worker_startup_states"}
    assert set(result["pod_states"].keys()) == {"0", "1"}
    assert result["pod_states"]["0"]["ready_workers"] == 4
    assert result["pod_states"]["1"]["degraded_workers"] == 2
    assert result["worker_startup_states"] == {
        "w-0": "ready",
        "w-1": "waiting_for_dataset",
    }


@pytest.mark.asyncio
async def test_empty_state_yields_empty_maps() -> None:
    """Brand-new controller with no pod-states yet — handler still returns
    a valid dict instead of raising."""
    instance = SimpleNamespace(_pod_states={}, _worker_startup_states={})
    result = await SystemControllerQueryMixin._on_get_pod_states(instance, _command())
    assert result == {"pod_states": {}, "worker_startup_states": {}}


@pytest.mark.asyncio
async def test_payload_round_trips_through_orjson() -> None:
    """The dispatcher orjson-encodes the return value into CommandOk.payload.
    Confirm that round-trip preserves the shape API-side decoders rely on."""
    instance = SimpleNamespace(
        _pod_states={"0": _pod("0", declared=2, ready=2)},
        _worker_startup_states={"w-0": "ready"},
    )
    result = await SystemControllerQueryMixin._on_get_pod_states(instance, _command())

    encoded = orjson.dumps(result)
    decoded = orjson.loads(encoded)
    assert decoded["pod_states"]["0"]["ready_workers"] == 2
    assert decoded["worker_startup_states"] == {"w-0": "ready"}

    # API-side path also re-validates each pod_states entry through
    # WorkerPodStateMessage(**raw); the model_dump emits the msgspec tag
    # field which the constructor does not accept, so strip it first
    # (matches what _aggregate_from_payload does).
    raw = {k: v for k, v in decoded["pod_states"]["0"].items() if k != "message_type"}
    rebuilt = WorkerPodStateMessage(**raw)
    assert rebuilt.ready_workers == 2
    assert rebuilt.pod_index == "0"
