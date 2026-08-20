# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime capability tests."""

import pytest

from aiperf.config.runtime import RuntimeConfig
from aiperf.plugin.enums import ServiceRunType


@pytest.mark.parametrize(
    ("service_run_type", "uses_worker_group_manager"),
    [
        pytest.param(
            ServiceRunType.MULTIPROCESSING,
            False,
            id="multiprocessing-spawns-workers-directly",
        ),
        pytest.param(
            ServiceRunType.KUBERNETES,
            True,
            id="kubernetes-routes-through-group-manager",
        ),
    ],
)
def test_uses_worker_group_manager_matches_runtime_topology(
    service_run_type: ServiceRunType,
    uses_worker_group_manager: bool,
) -> None:
    runtime = RuntimeConfig(service_run_type=service_run_type)

    assert runtime.uses_worker_group_manager is uses_worker_group_manager
