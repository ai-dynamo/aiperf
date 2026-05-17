# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import LifecycleState, ServiceCapability
from aiperf.common.messages import RegisterServiceCommand
from aiperf.plugin.enums import ServiceType


def _make(**overrides) -> RegisterServiceCommand:
    base = dict(
        command_id="cmd-1",
        service_id="svc-1",
        service_type=ServiceType.SYSTEM_CONTROLLER,
        state=LifecycleState.RUNNING,
    )
    base.update(overrides)
    return RegisterServiceCommand(**base)


def test_capabilities_default_empty_tuple() -> None:
    cmd = _make()
    assert cmd.capabilities == ()


def test_capabilities_round_trip_with_baseline() -> None:
    cmd = _make(capabilities=(ServiceCapability.BASELINE_COLLECTOR,))
    assert cmd.capabilities == (ServiceCapability.BASELINE_COLLECTOR,)
    payload = cmd.model_dump_json()
    parsed = RegisterServiceCommand.model_validate_json(payload)
    assert parsed.capabilities == (ServiceCapability.BASELINE_COLLECTOR,)


def test_capabilities_accepts_tuple_of_strings() -> None:
    cmd = _make(capabilities=("baseline_collector",))
    assert cmd.capabilities == ("baseline_collector",)
