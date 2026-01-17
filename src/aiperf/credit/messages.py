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
    """

    credit: Credit
    cancelled: bool = False
    first_token_sent: bool = False
    error: str | None = None


class FirstToken(Struct, frozen=True, kw_only=True, tag_field="t", tag="ft"):
    """Worker reports first token received (TTFT event).

    Sent by worker to router when first valid token is received from inference server.
    Router forwards to timing manager to release prefill concurrency slot.

    Attributes:
        credit_id: ID of the credit this TTFT is for.
        phase: Credit phase for routing to correct phase tracker.
        ttft_ns: Time to first token in nanoseconds (duration from request start).
    """

    credit_id: int
    phase: CreditPhase
    ttft_ns: int


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


# Union type for decoding router -> worker messages
# Credit is sent directly (no wrapper), CancelCredits for cancellation
RouterToWorkerMessage: TypeAlias = Credit | CancelCredits
