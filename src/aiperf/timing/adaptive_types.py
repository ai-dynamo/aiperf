# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared adaptive timing type aliases."""

from __future__ import annotations

from typing import Literal

AdaptiveControlVariable = Literal[
    "concurrency", "prefill_concurrency", "request_rate", "users"
]
