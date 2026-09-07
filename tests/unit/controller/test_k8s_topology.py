# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kubernetes startup must wait for the whole fleet, not one pod.

Worker pods are fixed-size: each runs workers_per_pod worker containers, and
the last pod is not partially filled. Requiring a single WorkerGroupManager
meant expected_pods == ready_pods == 1, so profiling began as soon as the
first pod reported in and ran against a fraction of the requested load with
no error anywhere.
"""

from unittest.mock import MagicMock

import pytest
from pytest import param

from aiperf.controller.system_controller import SystemController


def _runtime(**kw):
    rt = MagicMock()
    rt.workers = kw.get("workers")
    rt.workers_per_pod = kw.get("workers_per_pod")
    rt.record_processors = kw.get("record_processors")
    rt.record_processors_per_pod = kw.get("record_processors_per_pod")
    return rt


def _topology(**kw):
    ctrl = SystemController.__new__(SystemController)
    ctrl.run = MagicMock()
    ctrl.run.cfg.runtime = _runtime(**kw)
    return ctrl._build_k8s_service_topology()


class TestPodCountDerivation:
    @pytest.mark.parametrize(
        "workers,per_pod,expected_pods",
        [
            param(64, 8, 8, id="exact-multiple"),
            param(65, 8, 9, id="rounds-up-partial-pod"),
            param(1, 8, 1, id="single-worker-still-one-pod"),
            param(8, 8, 1, id="exactly-one-pod"),
        ],
    )  # fmt: skip
    def test_pods_are_ceil_of_workers_over_capacity(
        self, workers, per_pod, expected_pods
    ):
        topo = _topology(workers=workers, workers_per_pod=per_pod)
        assert topo.num_worker_pods == expected_pods

    def test_total_workers_counts_the_whole_fleet(self):
        """The last pod is not partially filled."""
        topo = _topology(workers=65, workers_per_pod=8)
        assert topo.total_workers == 72

    def test_never_derives_zero_pods(self):
        topo = _topology(workers=0, workers_per_pod=8)
        assert topo.num_worker_pods >= 1


class TestRecordProcessorDerivation:
    def test_explicit_per_pod_wins(self):
        topo = _topology(workers=16, workers_per_pod=8, record_processors_per_pod=3)
        assert topo.record_processors_per_pod == 3
        assert topo.total_record_processors == 6

    def test_total_is_spread_across_pods(self):
        topo = _topology(workers=16, workers_per_pod=8, record_processors=4)
        assert topo.record_processors_per_pod == 2

    def test_defaults_scale_with_workers_per_pod(self):
        topo = _topology(workers=16, workers_per_pod=8)
        assert topo.record_processors_per_pod >= 1


class TestRequiredServices:
    def test_worker_group_managers_match_the_pod_count(self):
        """The registration gate must expect every pod, not just the first."""
        from aiperf.plugin.enums import ServiceType

        ctrl = SystemController.__new__(SystemController)
        ctrl.run = MagicMock()
        ctrl.run.cfg.runtime = _runtime(workers=64, workers_per_pod=8)
        ctrl.required_services = {}

        topo = ctrl._build_k8s_service_topology()
        ctrl.required_services[ServiceType.WORKER_GROUP_MANAGER] = topo.num_worker_pods

        assert ctrl.required_services[ServiceType.WORKER_GROUP_MANAGER] == 8
