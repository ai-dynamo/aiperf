# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import msgspec
import pytest
from pytest import param

from aiperf.common.control_structs import (
    Command,
    CommandAck,
    CommandErr,
    CommandOk,
    CommandUnhandled,
    ControllerBoundMessage,
    Heartbeat,
    Registration,
    RegistrationAck,
    ServiceBoundMessage,
    StatusUpdate,
)

CONTROLLER_BOUND_DECODER = msgspec.msgpack.Decoder(ControllerBoundMessage)
SERVICE_BOUND_DECODER = msgspec.msgpack.Decoder(ServiceBoundMessage)
ENCODER = msgspec.msgpack.Encoder()


@pytest.mark.parametrize(
    "struct",
    [
        param(
            Registration(sid="s1", rid="r1", stype="worker", state="running"),
            id="registration_minimal",
        ),
        param(
            Registration(
                sid="s1",
                rid="r1",
                stype="worker",
                state="running",
                pod_name="aiperf-worker-0",
                pod_index="0",
                capabilities=("result_producer:telemetry",),
            ),
            id="registration_full",
        ),
        param(Heartbeat(sid="s1", stype="worker", state="running"), id="heartbeat"),
        param(StatusUpdate(sid="s1", stype="worker", state="stopping"), id="status"),
        param(Command(cid="c1", cmd="profile_start"), id="command_no_payload"),
        param(
            Command(cid="c1", cmd="profile_complete", payload=b'{"start_ns":1}'),
            id="command_with_payload",
        ),
        param(CommandAck(cid="c1", cmd="shutdown", sid="s1"), id="ack"),
        param(
            CommandOk(cid="c1", cmd="get_pod_states", sid="s1", payload=b"{}"),
            id="ok",
        ),
        param(
            CommandErr(cid="c1", cmd="profile_start", sid="s1", error="boom", traceback="tb"),
            id="err",
        ),
        param(
            CommandUnhandled(cid="c1", cmd="finalize_artifacts", sid="s1"),
            id="unhandled",
        ),
    ],
)  # fmt: skip
def test_controller_bound_struct_roundtrips_through_union(struct) -> None:
    assert CONTROLLER_BOUND_DECODER.decode(ENCODER.encode(struct)) == struct


@pytest.mark.parametrize(
    "struct",
    [
        param(RegistrationAck(rid="r1"), id="registration_ack"),
        param(Command(cid="c1", cmd="shutdown"), id="command"),
        param(CommandAck(cid="c1", cmd="spawn_workers", sid="ctl"), id="ack"),
        param(CommandOk(cid="c1", cmd="spawn_workers", sid="ctl", payload=b"1"), id="ok"),
        param(CommandErr(cid="c1", cmd="spawn_workers", sid="ctl", error="x"), id="err"),
        param(CommandUnhandled(cid="c1", cmd="nope", sid="ctl"), id="unhandled"),
    ],
)  # fmt: skip
def test_service_bound_struct_roundtrips_through_union(struct) -> None:
    assert SERVICE_BOUND_DECODER.decode(ENCODER.encode(struct)) == struct


def test_registration_ack_is_not_controller_bound() -> None:
    """RegistrationAck only travels controller->service; decoding it as
    controller-bound must fail rather than silently mis-tag."""
    with pytest.raises(msgspec.ValidationError):
        CONTROLLER_BOUND_DECODER.decode(ENCODER.encode(RegistrationAck(rid="r1")))


def test_control_structs_carry_no_timestamp_field() -> None:
    """Invariant I1: last-seen is stamped on receipt by the controller's clock.
    A wire timestamp would let a clock-skewed pod be reaped immediately."""
    for struct_type in (Registration, Heartbeat, StatusUpdate):
        names = {f.name for f in msgspec.structs.fields(struct_type)}
        assert not {n for n in names if n.endswith("_ns")}, struct_type.__name__


def test_command_response_structs_carry_cmd_for_identity_checks() -> None:
    """Invariant I8: _finalize_artifacts matches on (cmd, sid)."""
    for struct_type in (CommandAck, CommandOk, CommandErr, CommandUnhandled):
        names = {f.name for f in msgspec.structs.fields(struct_type)}
        assert "cmd" in names and "sid" in names, struct_type.__name__
