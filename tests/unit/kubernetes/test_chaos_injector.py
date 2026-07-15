# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess

import pytest

from tests.kubernetes.chaos.chaos_injector import ChaosInjector


class _KubectlPodAppears:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, *args: str, check: bool = True):  # noqa: ANN202
        self.calls += 1
        stdout = "" if self.calls == 1 else "controller-pod-0"
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")


@pytest.mark.asyncio
async def test_get_controller_pod_name_waits_until_pod_exists() -> None:
    kubectl = _KubectlPodAppears()
    injector = ChaosInjector(kubectl=kubectl)  # type: ignore[arg-type]

    pod_name = await injector.get_controller_pod_name(
        "aiperf-jobs", "chaos-c16", timeout=1.0
    )

    assert pod_name == "controller-pod-0"
    assert kubectl.calls == 2
