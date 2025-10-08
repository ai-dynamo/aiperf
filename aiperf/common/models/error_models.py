# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from aiohttp.client import ClientOSError
from pydantic import Field

from aiperf.common.exceptions import LifecycleOperationError
from aiperf.common.models.base_models import AIPerfBaseModel


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

    @classmethod
    def from_exception(cls, e: BaseException) -> "ErrorDetails":
        """Create an error details object from an exception."""
        code = e.errno if isinstance(e, ClientOSError) else None
        return cls(
            code=code,
            type=e.__class__.__name__,
            message=str(e),
        )


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
