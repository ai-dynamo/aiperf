# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import re
from typing import Any

from pydantic import Field

from aiperf.common.exceptions import LifecycleOperationError
from aiperf.common.models.base_models import AIPerfBaseModel

# Pre-compiled regex patterns for credential redaction
_REDACTIONS = [
    (re.compile(r"(?i)(authorization:\s*bearer\s+)[^\s,;]+"), r"\1***"),
    (re.compile(r"(?i)\b(api[-_ ]?key|token|secret)\s*=\s*[^&\s]+"), r"\1=***"),
    (re.compile(r"(?i)(x-api-key:\s*)[^\s,;]+"), r"\1***"),
]


class ErrorDetails(AIPerfBaseModel):
    """Encapsulates details about an error."""

    code: int | None = Field(
        default=None,
        description="The error code.",
    )
    type: str | None = Field(
        default=None,
        description="The error type.",
    )
    message: str = Field(
        ...,
        description="The error message.",
    )
    cause: str | None = Field(
        default=None,
        description="The cause of the error.",
    )
    cause_chain: list[str] | None = Field(
        default=None,
        description="List of exception type names in the __cause__ chain.",
    )
    details: Any | None = Field(
        default=None,
        description="Additional details about the error.",
    )

    @staticmethod
    def _safe_repr(value: Any, max_len: int = 4096) -> str:
        s = repr(value)
        for pattern, repl in _REDACTIONS:
            s = pattern.sub(repl, s)
        return s[:max_len] + "…" if len(s) > max_len else s

    def __eq__(self, other: Any) -> bool:
        """Check if the error details are equal by comparing the code, type, and message."""
        if not isinstance(other, ErrorDetails):
            return False
        return (
            self.code == other.code
            and self.type == other.type
            and self.message == other.message
        )

    def __hash__(self) -> int:
        """Hash the error details by hashing the code, type, and message."""
        return hash((self.code, self.type, self.message))

    @staticmethod
    def _build_cause_chain(e: BaseException) -> list[str] | None:
        """Build list of exception type names from __cause__ chain."""
        chain = []
        seen: set[int] = set()
        exc: BaseException | None = e
        while exc is not None and id(exc) not in seen:
            chain.append(exc.__class__.__name__)
            seen.add(id(exc))
            exc = exc.__cause__
        return chain if chain else None

    @classmethod
    def from_exception(cls, e: BaseException, **kwargs: Any) -> "ErrorDetails":
        """Create an error details object from an exception.

        Args:
            e: The exception to create error details from.
            **kwargs: Additional key-value pairs to include in details.
                Values that are None are filtered out.
        """
        details = {k: v for k, v in kwargs.items() if v is not None} or None

        error_details = cls(
            type=e.__class__.__name__,
            message=cls._safe_repr(e),
            cause=cls._safe_repr(e.__cause__) if e.__cause__ else None,
            cause_chain=cls._build_cause_chain(e),
            details=details,
        )
        if hasattr(e, "error_code") and isinstance(e.error_code, int):
            error_details.code = e.error_code
        return error_details


class ExitErrorInfo(AIPerfBaseModel):
    """Information about an error that should cause the process to exit."""

    error_details: ErrorDetails
    operation: str = Field(
        ...,
        description="The operation that caused the error.",
    )
    service_id: str | None = Field(
        default=None,
        description="The ID of the service that caused the error. If None, the error is not specific to a service.",
    )

    @classmethod
    def from_lifecycle_operation_error(
        cls, e: LifecycleOperationError
    ) -> "ExitErrorInfo":
        return cls(
            error_details=ErrorDetails.from_exception(e.original_exception or e),
            operation=e.operation,
            service_id=e.lifecycle_id,
        )


class ErrorDetailsCount(AIPerfBaseModel):
    """Count of error details."""

    error_details: ErrorDetails
    count: int = Field(
        ...,
        description="The count of the error details.",
    )
