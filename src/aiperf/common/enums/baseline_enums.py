# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Enums for best-effort phase baseline capture.

BaselineKind tags whether a baseline reading is taken near phase start (START)
or after credits have drained (END).

ServiceCapability is a generic capability tag advertised by services in their
RegisterServiceCommand.
"""

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class BaselineKind(CaseInsensitiveStrEnum):
    """Direction of a baseline reading relative to a phase."""

    START = "start"
    END = "end"


class ServiceCapability(CaseInsensitiveStrEnum):
    """Capability tags a service may advertise at registration time."""

    RESULT_PRODUCER = "result_producer"


_RESULT_PRODUCER_PREFIX = f"{ServiceCapability.RESULT_PRODUCER}:"


def make_result_producer_capability(domain: str) -> str:
    """Build a result-producer capability tag for a result domain."""

    return f"{_RESULT_PRODUCER_PREFIX}{domain}"


def parse_result_producer_capability(capability: str) -> str | None:
    """Return the result domain if capability is a result-producer tag."""

    if not capability.startswith(_RESULT_PRODUCER_PREFIX):
        return None
    domain = capability.removeprefix(_RESULT_PRODUCER_PREFIX)
    return domain or None
