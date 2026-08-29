# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native msgspec structs for the DEALER/ROUTER control channel.

All over-the-wire structs use tag_field="t" for efficient polymorphic decoding via tagged unions.
Tag values are short strings for minimal wire overhead.

Service -> Controller (ControllerBoundMessage):
    Registration ("reg") - service registration / connection probe, expects RegistrationAck back
    Heartbeat ("hb")     - periodic heartbeat, fire-and-forget
    StatusUpdate ("su")  - state change notification, fire-and-forget

Bidirectional (both unions):
    Command ("cmd")         - command request (controller->service or service->controller)
    CommandAck ("ca")       - acknowledged, no data
    CommandOk ("co")        - success with optional payload
    CommandErr ("ce")       - failure with error message
    CommandUnhandled ("cu") - no handler matched the command

Controller -> Service (ServiceBoundMessage):
    RegistrationAck ("ack") - response to Registration
"""

from typing import Any, TypeAlias

import orjson
from msgspec import Struct
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Service -> Controller: status
# ---------------------------------------------------------------------------


class Registration(Struct, frozen=True, kw_only=True, tag_field="t", tag="reg"):
    """Service registration / connection probe. Expects RegistrationAck back.

    In Kubernetes mode, services populate pod_name and pod_index from their
    environment for controller visibility.
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

    capabilities: tuple[str, ...] = ()
    """Result-producer capability tags advertised at registration.

    Parsed by ``parse_result_producer_capability`` on the controller and joined
    into the shutdown barrier via ``ResultJoinCoordinator.register``.
    """


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

    cmd: str = ""
    """CommandType string this response answers, for caller-side identity checks."""

    sid: str = ""
    """Service identifier of the acknowledging service."""


class CommandOk(Struct, frozen=True, kw_only=True, tag_field="t", tag="co"):
    """Command succeeded with optional result payload."""

    cid: str
    """Command identifier for the successful command."""

    cmd: str = ""
    """CommandType string this response answers, for caller-side identity checks."""

    sid: str = ""
    """Service identifier of the responding service."""

    payload: bytes = b""
    """Optional orjson-encoded result payload."""


class CommandErr(Struct, frozen=True, kw_only=True, tag_field="t", tag="ce"):
    """Command failed."""

    cid: str
    """Command identifier for the failed command."""

    cmd: str = ""
    """CommandType string this response answers, for caller-side identity checks."""

    sid: str = ""
    """Service identifier of the responding service."""

    error: str = ""
    """Human-readable error message describing the failure."""

    traceback: str = ""
    """Python traceback string for debugging, when available."""


class CommandUnhandled(Struct, frozen=True, kw_only=True, tag_field="t", tag="cu"):
    """No @on_command hook matched the command.

    Distinct from :class:`CommandAck` because callers treat "the service does
    not implement this command" as a failure (see ``_finalize_artifacts``)
    while a plain ack is success.
    """

    cid: str
    """Command identifier for the unhandled command."""

    cmd: str = ""
    """CommandType string that had no handler."""

    sid: str = ""
    """Service identifier that had no handler for the command."""


CommandResponse: TypeAlias = CommandAck | CommandOk | CommandErr | CommandUnhandled


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
    | Command
    | CommandAck
    | CommandOk
    | CommandErr
    | CommandUnhandled
)

ServiceBoundMessage: TypeAlias = (
    RegistrationAck | Command | CommandAck | CommandOk | CommandErr | CommandUnhandled
)


def encode_command_payload(result: Any) -> bytes:
    """Encode an ``@on_command`` hook result into a ``CommandOk`` payload.

    Shared by both dispatchers -- the controller's ``_dispatch_control_command``
    and the service's ``_execute_control_command`` -- so that a command answered
    in one direction encodes exactly as the same value would in the other.

    Ordering note: ``bytes`` and ``BaseModel`` are disjoint, so the two branches
    cannot both match and their relative order is not load-bearing. The previous
    duplicate implementations differed in that order, which made them look
    meaningfully different when they were not; the real divergence was
    ``isinstance(result, BaseModel)`` versus ``hasattr(result, "model_dump_json")``.
    The isinstance check is kept: it will not silently pick up an unrelated
    object that happens to expose a same-named method.
    """
    if isinstance(result, bytes):
        return result
    if isinstance(result, BaseModel):
        return result.model_dump_json().encode()
    return orjson.dumps(result)
