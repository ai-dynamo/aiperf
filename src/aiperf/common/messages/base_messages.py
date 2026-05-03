# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from typing import ClassVar

import orjson
from pydantic import Field

from aiperf.common.enums import MessageType
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.models.error_models import ErrorDetails
from aiperf.common.types import MessageTypeT


class Message(AIPerfBaseModel):
    """Base message class with automatic routing by message_type.

    Uses AutoRoutedModel for high-performance single-parse JSON deserialization
    with zero-copy dict routing. Supports nested discriminators (e.g., CommandMessage
    routes by 'command' field).

    Each message model should inherit from this class, set the message_type field,
    and define its own additional fields.
    """

    discriminator_field: ClassVar[str] = "message_type"

    message_type: MessageTypeT = Field(
        ...,
        description="The type of the message. Must be set in the subclass.",
    )

    request_ns: int | None = Field(
        default=None,
        description="Timestamp of the request",
    )

    request_id: str | None = Field(
        default=None,
        description="ID of the request",
    )

    def __str__(self) -> str:
        return self.model_dump_json(exclude_none=True)

    def to_json_bytes(self) -> bytes:
        """Serialize message to JSON bytes using orjson for optimal performance.

        This method uses orjson for high-performance serialization (6x faster for
        large records >20KB). It automatically excludes None fields to minimize
        message size.

        Returns:
            bytes: JSON-encoded message as bytes

        Note:
            Prefer this method over model_dump_json() for ZMQ message passing
            and other high-throughput scenarios.
        """
        return orjson.dumps(self.model_dump(exclude_none=True, mode="json"))


class RequiresRequestNSMixin(Message):
    """Mixin for messages that require a request_ns field."""

    request_ns: int = Field(  # type: ignore[assignment]
        default_factory=time.time_ns,
        description="Timestamp of the request in nanoseconds",
    )


class ErrorMessage(Message):
    """Message containing error data."""

    message_type: MessageTypeT = MessageType.ERROR

    error: ErrorDetails = Field(..., description="Error information")
