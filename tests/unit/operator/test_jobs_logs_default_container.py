# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Default container selection for operator log requests."""

from types import SimpleNamespace

from aiperf.operator.routers.jobs_logs import _default_container

from aiperf.kubernetes.constants import Containers


def _pod(names: list[str], *, annotated: str | None = None) -> SimpleNamespace:
    annotations = (
        {"kubectl.kubernetes.io/default-container": annotated} if annotated else {}
    )
    return SimpleNamespace(
        metadata=SimpleNamespace(annotations=annotations),
        spec=SimpleNamespace(containers=[SimpleNamespace(name=name) for name in names]),
    )


def test_controller_pod_defaults_to_control_plane() -> None:
    pod = _pod(
        [
            Containers.EVENT_BUS_PROXY,
            Containers.CONTROL_PLANE,
            Containers.RESULTS_SIDECAR,
        ]
    )

    assert _default_container(pod) == Containers.CONTROL_PLANE


def test_worker_pod_skips_event_bus_proxy() -> None:
    pod = _pod([Containers.EVENT_BUS_PROXY, Containers.WORKER_GROUP_MANAGER])

    assert _default_container(pod) == Containers.WORKER_GROUP_MANAGER


def test_default_container_annotation_wins() -> None:
    pod = _pod(
        [Containers.CONTROL_PLANE, Containers.RECORDS_MANAGER],
        annotated=Containers.RECORDS_MANAGER,
    )

    assert _default_container(pod) == Containers.RECORDS_MANAGER


def test_sidecar_only_pod_falls_back_to_first_container() -> None:
    pod = _pod([Containers.EVENT_BUS_PROXY, Containers.RESULTS_SIDECAR])

    assert _default_container(pod) == Containers.EVENT_BUS_PROXY
