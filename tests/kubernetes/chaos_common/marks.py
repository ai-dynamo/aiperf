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

When KIND_HAS_CILIUM is unset, the test still runs but is reported xfail;
because ``strict=True``, an unexpected pass is reported as a failure. When
KIND_HAS_CILIUM is set, the condition is False so the marker is inert and a
failure on the Cilium cluster is reported loudly as a plain failure.
"""
