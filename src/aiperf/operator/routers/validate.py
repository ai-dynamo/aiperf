# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Manifest validation endpoint — POST /api/v1/validate."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from aiperf.kubernetes.validate import validate_manifest

logger = logging.getLogger("aiperf.operator.ui")


class ValidateRequest(BaseModel):
    """Body of POST /api/v1/validate."""

    model_config = ConfigDict(extra="forbid")

    manifest: dict[str, Any] = Field(
        description="Full AIPerfJob or AIPerfSweep manifest dict to validate."
    )
    strict: bool = Field(
        default=False,
        description="When True, unknown spec fields are treated as errors instead of warnings.",
    )


class ValidateResponse(BaseModel):
    """Response from POST /api/v1/validate."""

    model_config = ConfigDict(extra="forbid")

    passed: bool = Field(description="True when the manifest has no validation errors.")
    errors: list[str] = Field(
        description="Validation errors that must be resolved before deployment."
    )
    warnings: list[str] = Field(description="Non-fatal validation warnings.")


def create_validate_router() -> APIRouter:
    """Create the ``/api/v1/validate`` router."""
    router = APIRouter(prefix="/api/v1", tags=["validate"])

    @router.post("/validate", response_model=ValidateResponse)
    async def validate(body: ValidateRequest) -> ValidateResponse:
        # The route is unauthenticated by design and exists purely to turn a bad
        # manifest into structured errors, so a validator that raises must still
        # answer within the documented contract instead of surfacing an HTTP 500.
        try:
            result = validate_manifest(body.manifest, strict=body.strict)
        except Exception as e:
            logger.exception("Manifest validation raised on POST /api/v1/validate")
            return ValidateResponse(
                passed=False,
                errors=[f"Validation aborted: {type(e).__name__}: {e}"],
                warnings=[],
            )
        return ValidateResponse(
            passed=result.passed,
            errors=result.errors,
            warnings=result.warnings,
        )

    return router
