# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The controller pod's writable remote-dataset cache."""

from __future__ import annotations

from typing import Any

import pytest
from pytest import param

from aiperf.config.deployment import PodTemplateConfig
from aiperf.kubernetes.constants import Containers
from aiperf.kubernetes.jobset import AIPerfJobSetSpec
from aiperf.kubernetes.jobset_helpers import (
    REMOTE_CACHE_ENV,
    REMOTE_CACHE_MOUNT_PATH,
    REMOTE_CACHE_VOLUME,
    build_shared_volumes,
)


def _controller_pod(pod_template: PodTemplateConfig | None = None) -> dict[str, Any]:
    spec = AIPerfJobSetSpec(
        name="rc",
        namespace="ns",
        job_id="rc-1",
        image="aiperf:ci",
        pod_template=pod_template or PodTemplateConfig(),
    )
    manifest = spec.to_k8s_manifest()
    job = next(
        j for j in manifest["spec"]["replicatedJobs"] if j["name"] == "controller"
    )
    return job["template"]["spec"]["template"]["spec"]


def _container(pod: dict[str, Any], name: str) -> dict[str, Any]:
    return next(c for c in pod["containers"] if c["name"] == name)


def _env_values(container: dict[str, Any], name: str) -> list[object]:
    return [item.get("value") for item in container["env"] if item["name"] == name]


def _mount_paths(container: dict[str, Any]) -> list[str]:
    return [mount["mountPath"] for mount in container["volumeMounts"]]


@pytest.mark.parametrize(
    "name",
    [
        param(Containers.DATASET_MANAGER, id="dataset-manager"),
        param(Containers.CONTROL_PLANE, id="control-plane"),
    ],
)  # fmt: skip
def test_controller_container_gets_writable_remote_cache(name: str) -> None:
    pod = _controller_pod()
    container = _container(pod, name)
    assert {
        "name": REMOTE_CACHE_VOLUME,
        "mountPath": REMOTE_CACHE_MOUNT_PATH,
    } in container["volumeMounts"]
    assert _env_values(container, REMOTE_CACHE_ENV) == [REMOTE_CACHE_MOUNT_PATH]
    volume = next(v for v in pod["volumes"] if v["name"] == REMOTE_CACHE_VOLUME)
    assert volume == {"name": REMOTE_CACHE_VOLUME, "emptyDir": {}}


@pytest.mark.parametrize(
    "name",
    [
        param(Containers.TIMING_MANAGER, id="timing-manager"),
        param(Containers.RECORDS_MANAGER, id="records-manager"),
        param(Containers.API, id="api"),
    ],
)  # fmt: skip
def test_other_controller_containers_do_not_mount_remote_cache(name: str) -> None:
    container = _container(_controller_pod(), name)
    assert REMOTE_CACHE_MOUNT_PATH not in _mount_paths(container)
    assert _env_values(container, REMOTE_CACHE_ENV) == []


_CACHE_CONTAINERS = [
    param(Containers.DATASET_MANAGER, id="dataset-manager"),
    param(Containers.CONTROL_PLANE, id="control-plane"),
]


def _user_pvc_template(mount_path: str) -> PodTemplateConfig:
    return PodTemplateConfig(
        volumes=[
            {"name": "pvc-cache", "persistentVolumeClaim": {"claimName": "cache"}}
        ],
        volume_mounts=[{"name": "pvc-cache", "mountPath": mount_path}],
    )


@pytest.mark.parametrize("name", _CACHE_CONTAINERS)  # fmt: skip
def test_user_remote_cache_env_wins(name: str) -> None:
    template = PodTemplateConfig(
        env=[{"name": REMOTE_CACHE_ENV, "value": "/pvc/aiperf-remote"}]
    )
    container = _container(_controller_pod(template), name)
    assert _env_values(container, REMOTE_CACHE_ENV) == ["/pvc/aiperf-remote"]


@pytest.mark.parametrize("name", _CACHE_CONTAINERS)  # fmt: skip
@pytest.mark.parametrize(
    "user_path",
    [
        param(REMOTE_CACHE_MOUNT_PATH, id="exact"),
        param(f"{REMOTE_CACHE_MOUNT_PATH}/", id="trailing-slash"),
    ],
)  # fmt: skip
def test_user_mount_at_remote_cache_path_skips_builtin_mount(
    name: str, user_path: str
) -> None:
    container = _container(_controller_pod(_user_pvc_template(user_path)), name)
    mounts = [
        m
        for m in container["volumeMounts"]
        if m["mountPath"].rstrip("/") == REMOTE_CACHE_MOUNT_PATH
    ]
    assert mounts == [{"name": "pvc-cache", "mountPath": user_path}]


@pytest.mark.parametrize("name", _CACHE_CONTAINERS)  # fmt: skip
def test_user_mount_without_env_override_keeps_cache_env(name: str) -> None:
    container = _container(
        _controller_pod(_user_pvc_template(REMOTE_CACHE_MOUNT_PATH)), name
    )
    assert _env_values(container, REMOTE_CACHE_ENV) == [REMOTE_CACHE_MOUNT_PATH]


def test_build_shared_volumes_includes_remote_cache_empty_dir() -> None:
    volumes = build_shared_volumes("bench", PodTemplateConfig())
    assert {"name": REMOTE_CACHE_VOLUME, "emptyDir": {}} in volumes
