# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import time
from typing import ClassVar

import orjson
from pydantic import Field

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums.message_enums import MessageType
from aiperf.common.models.base_models import AIPerfBaseModel, exclude_if_none
from aiperf.common.models.error_models import ErrorDetails
from aiperf.common.types import MessageTypeT

_logger = AIPerfLogger(__name__)


@exclude_if_none("request_ns", "request_id")
class Message(AIPerfBaseModel):
    """Base message class for optimized message handling. Based on the AIPerfBaseModel class,
    so it supports @exclude_if_none decorator. see :class:`AIPerfBaseModel` for more details.

    This class provides a base for all messages, including common fields like message_type,
    request_ns, and request_id. It also supports optional field exclusion based on the
    @exclude_if_none decorator.

    Each message model should inherit from this class, set the message_type field,
    and define its own additional fields.

    Example:
    ```python
    @exclude_if_none("some_field")
    class ExampleMessage(Message):
        some_field: int | None = Field(default=None)
        other_field: int = Field(default=1)
    ```
    """

    _message_type_lookup: ClassVar[dict[MessageTypeT, type["Message"]]] = {}
    """Lookup table for message types to their corresponding message classes. This is used to automatically
    deserialize messages from JSON strings to their corresponding class type."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "message_type") and cls.message_type is not None:
            # Store concrete message classes in the lookup table
            cls._message_type_lookup[cls.message_type] = cls
            _logger.trace(f"Added {cls.message_type} to message type lookup")

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

    # TODO: Does this allow you to use model_validate_json and have it forward it to from_json? Need to test.
    @classmethod
    def __get_validators__(cls):
        yield cls.from_json

    @classmethod
    def from_json(cls, json_str: str | bytes | bytearray) -> "Message":
        """Deserialize a message from a JSON string, attempting to auto-detect the message type.
        NOTE: If you already know the message type, use the more performant :meth:`from_json_with_type` instead."""
        data = json.loads(json_str)
        message_type = data.get("message_type")
        if not message_type:
            raise ValueError(f"Missing message_type: {json_str}")

        # Use cached message type lookup
        message_class = cls._message_type_lookup[message_type]
        if not message_class:
            raise ValueError(f"Unknown message type: {message_type}")

        return message_class.model_validate(data)

    @classmethod
    def from_json_with_type(
        cls, message_type: MessageTypeT, json_str: str | bytes | bytearray
    ) -> "Message":
        """Deserialize a message from a JSON string with a specific message type.
        NOTE: This is more performant than :meth:`from_json` because it does not need to
        convert the JSON string to a dictionary first."""
        # Use cached message type lookup
        message_class = cls._message_type_lookup[message_type]
        if not message_class:
            raise ValueError(f"Unknown message type: {message_type}")
        return message_class.model_validate_json(json_str)

    def to_json(self) -> str:
        """Fast serialization without full validation"""
        return orjson.dumps(self.__dict__).decode("utf-8")


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
