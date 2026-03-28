# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TypeAlias

from msgspec import Struct
from pydantic import Field

from aiperf.common.enums import CreditPhase, MessageType
from aiperf.common.messages import BaseServiceMessage
from aiperf.common.models import CreditPhaseStats
from aiperf.common.types import MessageTypeT
from aiperf.credit.structs import Credit
from aiperf.timing.config import CreditPhaseConfig


class CreditPhasesConfiguredMessage(BaseServiceMessage):
    """Message for credit phases configured. Sent by the TimingManager to report that the credit phases have been configured."""

    message_type: MessageTypeT = MessageType.CREDIT_PHASES_CONFIGURED
    configs: list[CreditPhaseConfig] = Field(
        ..., description="The credit phase configs in order of execution"
    )


class CreditPhaseStartMessage(BaseServiceMessage):
    """Message for credit phase start. Sent by the TimingManager to report that a credit phase has started."""

    message_type: MessageTypeT = MessageType.CREDIT_PHASE_START
    stats: CreditPhaseStats = Field(..., description="The credit phase stats")
    config: CreditPhaseConfig = Field(..., description="The credit phase config")


class CreditPhaseProgressMessage(BaseServiceMessage):
    """Sent by the TimingManager to report the progress of a credit phase."""

    message_type: MessageTypeT = MessageType.CREDIT_PHASE_PROGRESS
    stats: CreditPhaseStats = Field(..., description="The credit phase stats")


class CreditPhaseSendingCompleteMessage(BaseServiceMessage):
    """Message for credit phase sending complete. Sent by the TimingManager to report that a credit phase has completed sending."""

    message_type: MessageTypeT = MessageType.CREDIT_PHASE_SENDING_COMPLETE
    stats: CreditPhaseStats = Field(..., description="The credit phase stats")


class CreditPhaseCompleteMessage(BaseServiceMessage):
    """Message for credit phase complete. Sent by the TimingManager to report that a credit phase has completed."""

    message_type: MessageTypeT = MessageType.CREDIT_PHASE_COMPLETE
    stats: CreditPhaseStats = Field(..., description="The credit phase stats")


class CreditsCompleteMessage(BaseServiceMessage):
    """Credits complete message sent by the TimingManager to the System controller to signify all Credit Phases
    have been completed."""

    message_type: MessageTypeT = MessageType.CREDITS_COMPLETE


# =============================================================================
# Worker -> Router Messages
# =============================================================================


class WorkerConnected(Struct, frozen=True, kw_only=True, tag_field="t", tag="wc"):
    """Worker announces that its return path is connected.

    Sent by worker after establishing the credit/return channels.
    Router tracks the worker as connected but does not route credits yet.
    """

    worker_id: str
    """Unique worker service identifier."""


class WorkerDispatchable(Struct, frozen=True, kw_only=True, tag_field="t", tag="wd"):
    """Worker announces readiness to receive routed credits.

    Sent by worker after startup gates complete. Router uses this to add the
    worker to the routing pool.
    """

    worker_id: str
    """Unique worker service identifier."""


class WorkerUndispatchable(
    Struct, omit_defaults=True, frozen=True, kw_only=True, tag_field="t", tag="wu"
):
    """Worker announces that it should be removed from routing.

    Sent by worker when it remains connected but must stop receiving new
    credits.
    """

    worker_id: str
    """Unique worker service identifier."""

    reason: str | None = None
    """Human-readable reason for becoming undispatchable."""


class WorkerShutdown(Struct, frozen=True, kw_only=True, tag_field="t", tag="ws"):
    """Worker announces graceful shutdown.

    Sent by worker before disconnecting.
    Router uses this to remove worker from load balancing pool.
    """

    worker_id: str
    """Unique worker service identifier."""


class CreditReturn(
    Struct, omit_defaults=True, frozen=True, kw_only=True, tag_field="t", tag="cr"
):
    """Worker returns a credit after processing.

    Sent by worker to router after completing (or failing/cancelling) a request.
    Router uses this to update load tracking and notify timing manager.
    """

    credit: Credit
    """The credit being returned."""

    cancelled: bool = False
    """True if the credit was cancelled before completion."""

    first_token_sent: bool = False
    """True if FirstToken was sent before this return; used to release prefill slot."""

    error: str | None = None
    """Error message if the request failed (None on success)."""

    worker_detached: bool = False
    """True if the router received this return after the worker had already shut down."""


class FirstToken(Struct, frozen=True, kw_only=True, tag_field="t", tag="ft"):
    """Worker reports first token received (TTFT event).

    Sent by worker to router when first valid token is received from inference server.
    Router forwards to timing manager to release prefill concurrency slot.
    """

    credit_id: int
    """ID of the credit this TTFT is for."""

    phase: CreditPhase
    """Credit phase for routing to correct phase tracker."""

    ttft_ns: int
    """Time to first token in nanoseconds (duration from request start)."""


# =============================================================================
# Time Synchronization Messages (pre-flight RTT measurement)
# =============================================================================


class TimePing(Struct, frozen=True, kw_only=True, tag_field="t", tag="tp"):
    """Worker requests RTT measurement from router.

    Sent during startup before WorkerDispatchable. Router echoes back as TimePong
    so the worker can measure round-trip time on the credit channel.
    """

    sequence: int
    """Probe sequence number."""

    sent_at_ns: int
    """Worker perf_counter timestamp when ping was sent (time.perf_counter_ns)."""


class TimePong(Struct, frozen=True, kw_only=True, tag_field="t", tag="tpo"):
    """Router echoes back a TimePing as TimePong."""

    sequence: int
    """Probe sequence number (echoed from TimePing)."""

    sent_at_ns: int
    """Original worker send timestamp (echoed from TimePing)."""


# =============================================================================
# Router -> Worker Messages (Credit Channel)
# =============================================================================


class CancelCredits(Struct, frozen=True, kw_only=True, tag_field="t", tag="cc"):
    """Router requests worker to cancel in-flight credits.

    Worker should cancel any pending requests for the specified credit IDs.
    """

    credit_ids: set[int]
    """Set of credit IDs to cancel."""


# =============================================================================
# Reconciliation Messages
# =============================================================================


class InFlightReconciliation(
    Struct, frozen=True, kw_only=True, tag_field="t", tag="ifr"
):
    """Router sends its view of in-flight credits for a worker.

    Sent periodically on the credit channel. The worker compares against
    its own state and responds with an InFlightReport on the return channel.
    Credits missing from the worker's report for two consecutive cycles
    are treated as orphaned.
    """

    credit_ids: frozenset[int]
    """Credit IDs the router believes are in-flight for this worker."""


class InFlightReport(Struct, frozen=True, kw_only=True, tag_field="t", tag="ifp"):
    """Worker reports which credits it actually has in-flight.

    Sent on the return channel in response to InFlightReconciliation.
    """

    credit_ids: frozenset[int]
    """Credit IDs the worker is currently processing."""


# =============================================================================
# Channel Union Types
# =============================================================================

# Credit channel (Router -> Worker): truly unidirectional
CreditChannelMessage: TypeAlias = (
    Credit | CancelCredits | TimePong | InFlightReconciliation
)

# Return channel (Worker -> Router): truly unidirectional
WorkerToRouterMessage: TypeAlias = (
    WorkerConnected
    | WorkerDispatchable
    | WorkerUndispatchable
    | WorkerShutdown
    | CreditReturn
    | FirstToken
    | TimePing
    | InFlightReport
)

RouterToWorkerMessage: TypeAlias = Credit | CancelCredits | TimePong
