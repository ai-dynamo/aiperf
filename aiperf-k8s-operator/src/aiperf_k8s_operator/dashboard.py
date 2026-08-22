# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small dashboard readiness surface owned by the standalone distribution."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/dashboard/health")
def dashboard_health() -> dict[str, str]:
    """Return dashboard process readiness without exposing benchmark data."""
    return {"status": "ok"}
