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
    ControlMessage,
    Heartbeat,
    Registration,
    RegistrationAck,
    ServerMetricsStatus,
    StatusUpdate,
    TelemetryStatus,
)


def test_registration_roundtrips_through_msgpack() -> None:
    msg = Registration(sid="worker-0", rid="req-1", stype="worker", state="running")
    raw = msgspec.msgpack.encode(msg)
    back = msgspec.msgpack.decode(raw, type=ControlMessage)
    assert isinstance(back, Registration)
    assert back.sid == "worker-0"
    assert back.rid == "req-1"
    assert back.stype == "worker"


@pytest.mark.parametrize(
    "cls,tag",
    [
        param(Registration, "reg", id="registration"),
        param(RegistrationAck, "ack", id="registration-ack"),
        param(Heartbeat, "hb", id="heartbeat"),
        param(StatusUpdate, "su", id="status-update"),
        param(TelemetryStatus, "ts", id="telemetry-status"),
        param(ServerMetricsStatus, "sm", id="server-metrics-status"),
        param(Command, "cmd", id="command"),
        param(CommandAck, "ca", id="command-ack"),
        param(CommandOk, "co", id="command-ok"),
        param(CommandErr, "ce", id="command-err"),
    ],
)  # fmt: skip
def test_tag_values_are_stable_wire_constants(cls: type, tag: str) -> None:
    assert cls.__struct_config__.tag == tag
    assert cls.__struct_config__.tag_field == "t"


def test_structs_are_frozen() -> None:
    msg = Heartbeat(sid="worker-0", stype="worker", state="running")
    with pytest.raises(AttributeError):
        msg.sid = "worker-1"


def test_control_message_union_covers_every_struct() -> None:
    for cls in (
        Registration,
        RegistrationAck,
        Heartbeat,
        StatusUpdate,
        TelemetryStatus,
        ServerMetricsStatus,
        Command,
        CommandAck,
        CommandOk,
        CommandErr,
    ):
        assert cls in ControlMessage.__args__


def test_registration_declared_capacity_properties() -> None:
    msg = Registration(
        sid="workers-0",
        rid="req-1",
        stype="worker_manager",
        state="running",
        num_workers=8,
        num_record_processors=2,
    )
    assert msg.declared_worker_capacity == 8
    assert msg.declared_record_processor_capacity == 2
