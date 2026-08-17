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


class WorkerReady(Struct, frozen=True, kw_only=True, tag_field="t", tag="wr"):
    """Worker announces readiness to receive credits.

    Sent by worker immediately after connecting to router.
    Router uses this to add worker to load balancing pool.
    """

    worker_id: str


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


class FirstToken(
    Struct, omit_defaults=True, frozen=True, kw_only=True, tag_field="t", tag="ft"
):
    """Worker reports first token received (TTFT event).

    Sent by worker to router when first valid token is received from inference server.
    Router forwards to timing manager to release prefill concurrency slot AND to the
    graph first-token observer (post-TTFT first-token anchoring).

    Attributes:
        credit_id: ID of the credit this TTFT is for.
        phase: Credit phase for routing to correct phase tracker.
        ttft_ns: Time to first token in nanoseconds (duration from request start).
        phase_index: Concrete phase instance index used with ``phase`` to build the
            runtime key that locates the registered phase handler and prefill slot.
        trace_id: Agent-graph trace instance this event addresses
            (``{template}::{nonce}``, e.g. ``t-1::3f2a...``);
            None for non-graph (template/DAG) dispatch. The graph first-token
            observer keys off this to identify the emitting node's trace.
        x_correlation_id: Conversation instance ID of the emitting credit; None
            when the worker did not stamp it (non-graph fast path).
        turn_index: 0-based turn index of the emitting credit; None when unset.
    """

    credit_id: int
    phase: CreditPhase
    ttft_ns: int
    phase_index: int | None = None
    trace_id: str | None = None
    x_correlation_id: str | None = None
    turn_index: int | None = None


# Union type for decoding worker -> router messages
WorkerToRouterMessage: TypeAlias = (
    WorkerReady | WorkerShutdown | CreditReturn | FirstToken
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


class GraphTraceEnd(
    Struct,
    frozen=True,
    kw_only=True,
    tag_field="t",
    tag="te",  # codespell:ignore te
):
    """Router notifies the sticky worker that a graph trace execution ended.

    Sent ONCE per instance when the strategy reaps a trace instance's
    dispatch adapter (all in-flight dispatches drained) or at phase teardown
    for retained adapters. The router closes the instance's sticky session
    (graph sessions key on ``trace_id``) before forwarding; the worker uses
    it to evict the trace's dynamic-content pool entries. Ids only -- no
    content.

    Attributes:
        trace_id: Agent-graph trace instance id (``{template}::{nonce}``).
    """

    trace_id: str


# Union type for decoding router -> worker messages: Credit is sent directly
# (no wrapper); CancelCredits for cancellation; GraphTraceEnd for graph trace
# lifecycle.
RouterToWorkerMessage: TypeAlias = Credit | CancelCredits | GraphTraceEnd
