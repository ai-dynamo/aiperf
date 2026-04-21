# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import ConfigDict

from aiperf.common.exceptions import LifecycleOperationError
from aiperf.common.redact import redact_string


@dataclass(slots=True, kw_only=True, eq=False)
class ErrorDetails:
    """Encapsulates details about an error.

    Slotted dataclass — shared type usable natively in both msgspec contexts
    (``ExitErrorInfo``, ``RequestRecord.error``, ``ErrorMessage.error``) and
    Pydantic contexts (transitively via ``ErrorDetailsCount`` in
    ``JsonExportData.error_summary``). ``__pydantic_config__`` lets Pydantic
    validate/serialize it natively without a compat shim.

    Equality and hash are defined over ``(code, type, message)`` so two
    errors with different stack traces but the same class/code dedup
    correctly inside ``ErrorDetailsCount``.
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    message: str
    code: int | None = None
    type: str | None = None
    cause: str | None = None
    cause_chain: list[str] | None = None
    details: Any | None = None

    @staticmethod
    def _safe_repr(value: Any, max_len: int = 4096) -> str:
        s = redact_string(repr(value))
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
    def _build_cause_chain(e: BaseException | None) -> list[str] | None:
        """Build list of exception type names from the exception chain.

        Follows both explicit chaining (__cause__, set by ``raise X from Y``)
        and implicit chaining (__context__, set when raising inside an except
        block). This is critical because libraries like ``transformers`` often
        re-raise without ``from``, so the root-cause type (e.g. GatedRepoError)
        only appears in __context__.
        """
        chain: list[str] = []
        seen: set[int] = set()
        exc: BaseException | None = e
        while exc is not None and id(exc) not in seen:
            chain.append(exc.__class__.__name__)
            seen.add(id(exc))
            if exc.__cause__ is not None:
                exc = exc.__cause__
            elif exc.__suppress_context__:
                break
            else:
                exc = exc.__context__
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

        code: int | None = None
        if hasattr(e, "error_code") and isinstance(e.error_code, int):
            code = e.error_code

        return cls(
            type=e.__class__.__name__,
            message=cls._safe_repr(e),
            cause=cls._safe_repr(e.__cause__) if e.__cause__ else None,
            cause_chain=cls._build_cause_chain(e),
            details=details,
            code=code,
        )


@dataclass(slots=True, kw_only=True, frozen=True)
class ExitErrorInfo:
    """Information about an error that should cause the process to exit."""

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    error_details: ErrorDetails
    operation: str
    service_id: str | None = None

    @classmethod
    def from_lifecycle_operation_error(
        cls, e: LifecycleOperationError
    ) -> "ExitErrorInfo":
        return cls(
            error_details=ErrorDetails.from_exception(e.original_exception or e),
            operation=e.operation,
            service_id=e.lifecycle_id,
        )


@dataclass(slots=True, kw_only=True, frozen=True)
class ErrorDetailsCount:
    """Count of error details. Shared type: lives in both msgspec contexts
    (``ProfileResults.error_summary``) and Pydantic export models
    (``JsonExportData.error_summary`` etc.) without a compat shim.
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    error_details: ErrorDetails
    count: int
