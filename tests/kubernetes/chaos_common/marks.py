# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pytest marks shared across the chaos_common/chaos_dynamo suites."""

from __future__ import annotations

import os

import pytest

cilium_on_kind_required = pytest.mark.xfail(
    condition=not os.environ.get("KIND_HAS_CILIUM"),
    reason=(
        "kindnet does not honor NetworkPolicy; set KIND_HAS_CILIUM=1 when "
        "running against a Cilium-equipped cluster (see chaos_common/README.md)."
    ),
    strict=True,
)
"""
Apply to tests that require a NetworkPolicy-aware CNI (currently only D704).

When KIND_HAS_CILIUM is unset, the test is xfail-skipped. When set, the
xfail constraint flips to strict - if the test fails on a Cilium cluster,
pytest reports it loudly.
"""
