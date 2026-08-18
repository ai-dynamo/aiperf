# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TypeAlias

from msgspec import Struct
from pydantic import Field

from aiperf.common.enums import CreditPhase, MessageType
from aiperf.common.messages import BaseServiceMessage
from aiperf.common.models import CreditPhaseStats
from aiperf.common.models.branch_stats import BranchStats
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
    branch_stats: BranchStats | None = Field(
        default=None,
        description="DAG branch orchestration counters at phase completion. "
        "None for non-DAG runs (no BranchOrchestrator); a populated "
        "BranchStats snapshot for DAG-shaped runs (FORK or SPAWN). "
        "RecordsManager forwards this to ProfileResults so the JSON "
        "exporter can splice it into profile_export_aiperf.json.",
    )


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

    Connectivity and dispatchability are deliberately separate: in
    Kubernetes the worker is connected long before its pod-local dataset
    exists, and a worker without a dataset cannot serve a credit. Routing on
    connectivity alone hands credits to a worker that fails every one of
    them, silently, and the RecordsManager barrier then waits forever for
    records that never arrive.
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

    NOT CURRENTLY SENT. StickyCreditRouter handles it, but no worker emits it:
    the third leg of the dispatchability protocol is wired end-to-end except
    for a trigger. Nothing in the worker today reaches "connected but must not
    be routed to" -- shutdown announces WorkerShutdown from @on_stop, which
    unregisters outright, and dataset readiness only ever latches forward via
    WorkerDispatchable.

    So a worker that degrades mid-run (loses its dataset, starts failing every
    credit) has only two options: keep failing routed credits, or die and wait
    roughly 3x STALE_TIME for evict_stale_workers to notice. Choosing the
    trigger condition is a product decision -- deliberately not invented here,
    because a wrong one silently stops routing to healthy workers.
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


class CreditReturn(
    Struct, omit_defaults=True, frozen=True, kw_only=True, tag_field="t", tag="cr"
):
    """Worker returns a credit after processing.

    Sent by worker to router after completing (or failing/cancelling) a request.
    Router uses this to update load tracking and notify timing manager.

    Attributes:
        credit: The credit being returned.
        cancelled: True if the credit was cancelled before completion.
        first_token_sent: True if FirstToken was sent before this return.
            Used by orchestrator to release prefill slot if not already released.
        error: Error message if the request failed (None on success).
        request_latency_ns: Request latency in nanoseconds using the same
            start/end semantics as the records-pipeline request_latency metric.
            None when the request did not produce a valid content response.
        inter_token_latency_ns: Inter-token latency in nanoseconds for adaptive SLA evaluation.
            None when the request does not have valid content timing and
            output sequence length.
        output_sequence_length: Output sequence length in tokens from usage
            data, when available.
        worker_id: Returning worker's id. Only stamped on the PUSH/PULL return
            channel (CommAddress.CREDIT_RETURN), where there is no ZMQ envelope
            identity; None on the ROUTER/DEALER path (identity comes from the
            envelope). Lets the router attribute the return to the right worker.
    """

    credit: Credit
    cancelled: bool = False
    first_token_sent: bool = False
    error: str | None = None
    request_latency_ns: int | None = None
    inter_token_latency_ns: float | None = None
    output_sequence_length: int | None = None
    worker_id: str | None = None


class FirstToken(Struct, frozen=True, kw_only=True, tag_field="t", tag="ft"):
    """Worker reports first token received (TTFT event).

    Sent by worker to router when first valid token is received from inference server.
    Router forwards to timing manager to release prefill concurrency slot.

    Attributes:
        credit_id: ID of the credit this TTFT is for.
        phase: Credit phase for routing to correct phase tracker.
        ttft_ns: Time to first token in nanoseconds (duration from request start).
        phase_index: Concrete phase instance index used with ``phase`` to build the
            runtime key that locates the registered phase handler and prefill slot.
    """

    credit_id: int
    phase: CreditPhase
    ttft_ns: int
    phase_index: int | None = None


# =============================================================================
# Time Synchronization Messages (pre-flight RTT measurement)
# =============================================================================


class TimePing(Struct, frozen=True, kw_only=True, tag_field="t", tag="tp"):
    """Worker requests RTT measurement from router.

    Sent during startup before the worker declares itself dispatchable. The router
    echoes it back as a TimePong so the worker can measure round-trip time on the
    credit channel itself, rather than on a separate socket with different queuing.

    Attributes:
        sequence: Probe sequence number.
        sent_at_ns: Worker perf_counter timestamp when the ping was sent
            (``time.perf_counter_ns``).
    """

    sequence: int
    sent_at_ns: int


class TimePong(Struct, frozen=True, kw_only=True, tag_field="t", tag="tpo"):
    """Router echoes back a TimePing as TimePong.

    Both fields are echoed verbatim so the worker can match the reply to its probe
    and compute RTT entirely against its own clock -- no router clock is involved,
    which is what makes the RTT measurement immune to cross-machine skew.

    Attributes:
        sequence: Probe sequence number (echoed from TimePing).
        sent_at_ns: Original worker send timestamp (echoed from TimePing).
    """

    sequence: int
    sent_at_ns: int


# Union type for decoding worker -> router messages
WorkerToRouterMessage: TypeAlias = (
    WorkerConnected
    | WorkerDispatchable
    | WorkerUndispatchable
    | WorkerShutdown
    | CreditReturn
    | FirstToken
    | TimePing
)

# =============================================================================
# Router -> Worker Messages
# =============================================================================


class CancelCredits(Struct, frozen=True, kw_only=True, tag_field="t", tag="cc"):
    """Router requests worker to cancel in-flight credits.

    Worker should cancel any pending requests for the specified credit IDs.

    Attributes:
        credit_ids: Set of credit IDs to cancel.
    """

    credit_ids: set[int]


# Union type for decoding router -> worker messages
# Credit is sent directly (no wrapper), CancelCredits for cancellation
RouterToWorkerMessage: TypeAlias = Credit | CancelCredits | TimePong
