# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native msgspec structs for the DEALER/ROUTER control channel.

All over-the-wire structs use tag_field="t" for efficient polymorphic decoding via tagged unions.
Tag values are short strings for minimal wire overhead.

Service -> Controller (ControllerBoundMessage):
    Registration ("reg")       - service registration / connection probe, expects RegistrationAck back
    Heartbeat ("hb")           - periodic heartbeat, fire-and-forget
    StatusUpdate ("su")        - state change notification, fire-and-forget
    MemoryReport ("mr")        - self-reported memory snapshot, fire-and-forget
    TelemetryStatus ("ts")     - telemetry availability from TelemetryManager, fire-and-forget
    ServerMetricsStatus ("sm") - server metrics availability from ServerMetricsManager, fire-and-forget

Bidirectional (both unions):
    Command ("cmd")   - command request (controller->service or service->controller)
    CommandAck ("ca") - acknowledged, no data
    CommandOk ("co")  - success with optional payload
    CommandErr ("ce") - failure with error message

Controller -> Service (ServiceBoundMessage):
    RegistrationAck ("ack") - response to Registration
"""

from typing import TypeAlias

from msgspec import Struct

# ---------------------------------------------------------------------------
# Service -> Controller: status & telemetry
# ---------------------------------------------------------------------------


class Registration(Struct, frozen=True, kw_only=True, tag_field="t", tag="reg"):
    """Service registration / connection probe. Expects RegistrationAck back.

    In Kubernetes mode, services populate pod_name and pod_index from their
    environment for controller visibility. Group-managed services may also
    declare child capacity so the controller can reason about aggregate worker
    availability without expecting each child service to register directly.
    """

    sid: str
    """Unique service identifier for the registering service."""

    rid: str
    """Request identifier used to correlate the RegistrationAck response."""

    stype: str
    """Service type string reported during registration."""

    state: str
    """Current lifecycle state of the registering service."""

    pod_name: str | None = None
    """Kubernetes pod name, populated in K8s mode."""

    pod_index: str | None = None
    """Kubernetes pod index within the StatefulSet or JobSet."""

    num_workers: int | None = None
    """Declared child worker capacity for group-managed services."""

    num_record_processors: int | None = None
    """Declared child record-processor capacity for group-managed services."""

    @property
    def declared_worker_capacity(self) -> int | None:
        """Return the declared child worker capacity for this group."""
        return self.num_workers

    @property
    def declared_record_processor_capacity(self) -> int | None:
        """Return the declared child record-processor capacity for this group."""
        return self.num_record_processors


class Heartbeat(Struct, frozen=True, kw_only=True, tag_field="t", tag="hb"):
    """Periodic heartbeat (fire-and-forget)."""

    sid: str
    """Unique service identifier for the heartbeat sender."""

    stype: str
    """Service type of the heartbeat sender."""

    state: str
    """Current lifecycle state of the heartbeat sender."""


class StatusUpdate(Struct, frozen=True, kw_only=True, tag_field="t", tag="su"):
    """State change notification (fire-and-forget)."""

    sid: str
    """Unique service identifier for the service reporting a state change."""

    stype: str
    """Service type of the service reporting a state change."""

    state: str
    """New lifecycle state being reported."""


class MemoryReport(Struct, frozen=True, kw_only=True, tag_field="t", tag="mr"):
    """Self-reported memory snapshot (fire-and-forget)."""

    sid: str
    """Unique service identifier for the reporting service."""

    stype: str
    """Service type of the reporting service."""

    pid: int
    """Operating system process ID of the reporting service."""

    phase: str
    """Benchmark phase during which this memory snapshot was captured."""

    pss_bytes: int
    """Proportional set size memory in bytes."""

    rss_bytes: int | None = None
    """Resident set size memory in bytes, when available."""

    uss_bytes: int | None = None
    """Unique set size memory in bytes, when available."""

    shared_bytes: int | None = None
    """Shared memory in bytes, when available."""


class TelemetryStatus(Struct, frozen=True, kw_only=True, tag_field="t", tag="ts"):
    """Telemetry availability status from TelemetryManager (fire-and-forget)."""

    sid: str
    """Unique service identifier for the TelemetryManager instance."""

    enabled: bool
    """Whether telemetry collection is currently enabled."""

    reason: str | None = None
    """Human-readable reason when telemetry is disabled."""

    endpoints_configured: tuple[str, ...] = ()
    """Telemetry endpoint URLs that were configured."""

    endpoints_reachable: tuple[str, ...] = ()
    """Telemetry endpoint URLs that responded successfully."""


class ServerMetricsStatus(Struct, frozen=True, kw_only=True, tag_field="t", tag="sm"):
    """Server metrics availability status from ServerMetricsManager (fire-and-forget)."""

    sid: str
    """Unique service identifier for the ServerMetricsManager instance."""

    enabled: bool
    """Whether server metrics collection is currently enabled."""

    reason: str | None = None
    """Human-readable reason when server metrics are disabled."""

    endpoints_configured: tuple[str, ...] = ()
    """Server metrics endpoint URLs that were configured."""

    endpoints_reachable: tuple[str, ...] = ()
    """Server metrics endpoint URLs that responded successfully."""


# ---------------------------------------------------------------------------
# Bidirectional: command request-reply
# ---------------------------------------------------------------------------


class Command(Struct, frozen=True, kw_only=True, tag_field="t", tag="cmd"):
    """Command request. Sent in either direction (controller->service or service->controller)."""

    cid: str
    """Command identifier used to correlate request and response."""

    cmd: str
    """CommandType string value identifying the command to execute."""

    payload: bytes = b""
    """Command-specific data encoded with orjson.dumps."""


class CommandAck(Struct, frozen=True, kw_only=True, tag_field="t", tag="ca"):
    """Command acknowledged, no result data."""

    cid: str
    """Command identifier being acknowledged."""

    sid: str = ""
    """Service identifier of the acknowledging service."""


class CommandOk(Struct, frozen=True, kw_only=True, tag_field="t", tag="co"):
    """Command succeeded with optional result payload."""

    cid: str
    """Command identifier for the successful command."""

    sid: str = ""
    """Service identifier of the responding service."""

    payload: bytes = b""
    """Optional orjson-encoded result payload."""


class CommandErr(Struct, frozen=True, kw_only=True, tag_field="t", tag="ce"):
    """Command failed."""

    cid: str
    """Command identifier for the failed command."""

    sid: str = ""
    """Service identifier of the responding service."""

    error: str = ""
    """Human-readable error message describing the failure."""

    traceback: str = ""
    """Python traceback string for debugging, when available."""


CommandResponse: TypeAlias = CommandAck | CommandOk | CommandErr


# ---------------------------------------------------------------------------
# Controller -> Service only
# ---------------------------------------------------------------------------


class RegistrationAck(Struct, frozen=True, kw_only=True, tag_field="t", tag="ack"):
    """Acknowledgement of a Registration."""

    rid: str
    """Request identifier copied from the triggering Registration."""


# ---------------------------------------------------------------------------
# Union types for polymorphic decoding
# ---------------------------------------------------------------------------

ControllerBoundMessage: TypeAlias = (
    Registration
    | Heartbeat
    | StatusUpdate
    | MemoryReport
    | TelemetryStatus
    | ServerMetricsStatus
    | Command
    | CommandAck
    | CommandOk
    | CommandErr
)

ServiceBoundMessage: TypeAlias = (
    RegistrationAck | Command | CommandAck | CommandOk | CommandErr
)
