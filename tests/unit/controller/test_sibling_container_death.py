# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A dead sibling container in the controller pod must abort, not hang.

The controller pod runs the control plane alongside dataset-manager,
timing-manager, records-manager and optional telemetry sidecars. If one of
those dies before it registers -- server-metrics-manager hitting its memory
limit is the observed case -- the configure wait blocks for the full
PROFILE_CONFIGURE_TIMEOUT and then reports a generic timeout, naming nothing.

The pod poll already fetches the controller pod; it was being discarded
because ``extract_pod_snapshot`` keeps only the ``workers`` replicated job.
"""

from types import SimpleNamespace

import pytest
from pytest import param

from aiperf.controller.kubernetes_pod_helpers import dead_sibling_containers
from aiperf.kubernetes.constants import Containers


def _cs(name: str, *, terminated: bool, reason: str = "Error", exit_code: int = 1):
    state = SimpleNamespace(
        terminated=SimpleNamespace(reason=reason, exit_code=exit_code)
        if terminated
        else None
    )
    return SimpleNamespace(name=name, state=state)


def _pod(name: str, replicated_job: str, container_statuses: list):
    return SimpleNamespace(
        metadata=SimpleNamespace(
            name=name,
            labels={"jobset.sigs.k8s.io/replicatedjob-name": replicated_job},
        ),
        status=SimpleNamespace(container_statuses=container_statuses),
    )


class TestDeadSiblingContainers:
    def test_detects_oomkilled_service_container(self):
        pods = [
            _pod(
                "job-controller-0-0-abcde",
                "controller",
                [
                    _cs(Containers.CONTROL_PLANE, terminated=False),
                    _cs(
                        Containers.SERVER_METRICS_MANAGER,
                        terminated=True,
                        reason="OOMKilled",
                        exit_code=137,
                    ),
                ],
            )
        ]
        dead = dead_sibling_containers(pods)
        assert dead == [(Containers.SERVER_METRICS_MANAGER, "OOMKilled", 137)]

    def test_ignores_worker_pods(self):
        """Worker pods have their own monitoring path; do not double-report."""
        pods = [
            _pod(
                "job-workers-0-0",
                "workers",
                [_cs(Containers.WORKER_GROUP_MANAGER, terminated=True)],
            )
        ]
        assert dead_sibling_containers(pods) == []

    @pytest.mark.parametrize(
        "container",
        [
            param(Containers.CONTROL_PLANE, id="control-plane"),
            param(Containers.EVENT_BUS_PROXY, id="event-bus-proxy"),
            param(Containers.RESULTS_SIDECAR, id="results-sidecar"),
        ],
    )  # fmt: skip
    def test_ignores_infrastructure_containers(self, container):
        """Infra containers are not aiperf services; the control plane is us."""
        pods = [
            _pod("job-controller-0-0", "controller", [_cs(container, terminated=True)])
        ]
        assert dead_sibling_containers(pods) == []

    def test_ignores_clean_exit(self):
        """A zero exit is an optional service finishing, not a failure."""
        pods = [
            _pod(
                "job-controller-0-0",
                "controller",
                [
                    _cs(
                        Containers.GPU_TELEMETRY_MANAGER,
                        terminated=True,
                        reason="Completed",
                        exit_code=0,
                    )
                ],
            )
        ]
        assert dead_sibling_containers(pods) == []

    def test_oomkilled_counts_even_with_zero_exit_code(self):
        """OOMKilled is a failure however the exit code reads."""
        pods = [
            _pod(
                "job-controller-0-0",
                "controller",
                [
                    _cs(
                        Containers.DATASET_MANAGER,
                        terminated=True,
                        reason="OOMKilled",
                        exit_code=0,
                    )
                ],
            )
        ]
        assert dead_sibling_containers(pods) == [
            (Containers.DATASET_MANAGER, "OOMKilled", 0)
        ]

    def test_running_containers_are_not_reported(self):
        pods = [
            _pod(
                "job-controller-0-0",
                "controller",
                [_cs(Containers.RECORDS_MANAGER, terminated=False)],
            )
        ]
        assert dead_sibling_containers(pods) == []

    def test_tolerates_missing_status_and_labels(self):
        """A pod mid-creation must not raise inside the monitor loop."""
        bare = SimpleNamespace(metadata=None, status=None)
        no_status = _pod("x", "controller", [])
        no_status.status = None
        assert dead_sibling_containers([bare, no_status]) == []


class TestMonitorLoopWiring:
    """A dead sibling must reach the abort event the controller already waits on."""

    def test_dead_sibling_sets_the_abort_event(self):
        import asyncio

        from aiperf.controller._pod_monitoring_mixin import PodMonitoringMixin

        mixin = PodMonitoringMixin.__new__(PodMonitoringMixin)
        mixin.pod_failure_abort_event = asyncio.Event()
        mixin.pod_failure_abort_reason = ""
        mixin.error = lambda *_a, **_k: None

        mixin._check_dead_sibling_containers(
            [
                _pod(
                    "job-controller-0-0",
                    "controller",
                    [
                        _cs(
                            Containers.SERVER_METRICS_MANAGER,
                            terminated=True,
                            reason="OOMKilled",
                            exit_code=137,
                        )
                    ],
                )
            ]
        )

        assert mixin.pod_failure_abort_event.is_set()
        assert "server-metrics-manager" in mixin.pod_failure_abort_reason
        assert "OOMKilled" in mixin.pod_failure_abort_reason

    def test_healthy_pod_leaves_the_event_clear(self):
        import asyncio

        from aiperf.controller._pod_monitoring_mixin import PodMonitoringMixin

        mixin = PodMonitoringMixin.__new__(PodMonitoringMixin)
        mixin.pod_failure_abort_event = asyncio.Event()
        mixin.pod_failure_abort_reason = ""
        mixin.error = lambda *_a, **_k: None

        mixin._check_dead_sibling_containers(
            [
                _pod(
                    "job-controller-0-0",
                    "controller",
                    [_cs(Containers.RECORDS_MANAGER, terminated=False)],
                )
            ]
        )
        assert not mixin.pod_failure_abort_event.is_set()
