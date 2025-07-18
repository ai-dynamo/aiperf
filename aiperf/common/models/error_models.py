# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from pydantic import Field

from aiperf.common.models import AIPerfBaseModel


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
    def from_exception(cls, e: Exception) -> "ErrorDetails":
        """Create an error details object from an exception."""
        return cls(
            type=e.__class__.__name__,
            message=str(e),
        )


class ErrorDetailsCount(AIPerfBaseModel):
    """Count of error details."""

    error_details: ErrorDetails
    count: int = Field(
        ...,
        description="The count of the error details.",
    )
