# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import time
from typing import Any, ClassVar, Literal

import orjson
from pydantic import (
    Field,
    model_serializer,
)

from aiperf.common.enums import (
    MessageType,
)
from aiperf.common.models import (
    ErrorDetails,
    ExcludeIfNoneMixin,
    exclude_if_none,
)


@exclude_if_none(["request_ns", "request_id"])
class Message(ExcludeIfNoneMixin):
    """Base message class for optimized message handling.

    This class provides a base for all messages, including common fields like message_type,
    request_ns, and request_id. It also supports optional field exclusion based on the
    @exclude_if_none decorator.

    Each message model should inherit from this class, set the message_type field,
    and define its own additional fields.
    Optionally, the @exclude_if_none decorator can be used to specify which fields
    should be excluded from the serialized message if they are None.

    Example:
    ```python
    @exclude_if_none(["some_field"])
    class ExampleMessage(Message):
        some_field: int | None = Field(default=None)
        other_field: int = Field(default=1)
    ```
    """

    _exclude_if_none_fields: ClassVar[set[str]] = set()
    """Set of field names that should be excluded from the serialized message if they
    are None. This is set by the @exclude_if_none decorator.
    """

    _message_type_lookup: ClassVar[dict[MessageType, type["Message"]]] = {}

    def __init_subclass__(cls, **kwargs: dict[str, Any]):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "message_type"):
            cls._message_type_lookup[cls.message_type] = cls

    message_type: MessageType | Any = Field(
        ...,
        description="Type of the message",
    )

    request_ns: int | None = Field(
        default=None,
        description="Timestamp of the request",
    )

    request_id: str | None = Field(
        default=None,
        description="ID of the request",
    )

    @model_serializer
    def _serialize_message(self) -> dict[str, Any]:
        """Serialize the message to a dictionary.

        This method overrides the default serializer to exclude fields that have a
        value of None and have the EXCLUDE_IF_NONE json_schema_extra key set to True.
        """
        return {
            k: v
            for k, v in self
            if not (k in self._exclude_if_none_fields and v is None)
        }

    @classmethod
    def __get_validators__(cls):
        yield cls.from_json

    @classmethod
    def from_json(cls, json_str: str | bytes | bytearray) -> "Message":
        """Fast deserialization without full validation"""
        data = json.loads(json_str)
        message_type = data.get("message_type")
        if not message_type:
            raise ValueError("Missing message_type")

        # Use cached message type lookup
        message_class = cls._message_type_lookup[message_type]
        if not message_class:
            raise ValueError(f"Unknown message type: {message_type}")

        return message_class(**data)

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

    message_type: Literal[MessageType.ERROR] = MessageType.ERROR

    error: ErrorDetails = Field(..., description="Error information")
