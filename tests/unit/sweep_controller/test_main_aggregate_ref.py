# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.kubernetes.jobset import controller_dns_name
from aiperf.kubernetes.results_artifacts import API_RESULTS_FILES_PATH
from aiperf.sweep_controller.main import (
    SWEEP_CONTROLLER_RESULTS_SIDECAR_PORT,
    sweep_aggregate_api_path,
    sweep_controller_host,
)


def test_sweep_controller_ref_uses_configured_sidecar_port() -> None:
    assert SWEEP_CONTROLLER_RESULTS_SIDECAR_PORT == K8sEnvironment.PORTS.RESULTS_SIDECAR


def test_sweep_controller_ref_uses_cluster_resolvable_host() -> None:
    assert sweep_controller_host("concurrency-sweep", "bench") == controller_dns_name(
        "aiperf-concurrency-sweep", "bench"
    )


def test_sweep_controller_ref_path_matches_served_aggregate(
    tmp_path: Path,
) -> None:
    aggregate = tmp_path / "bench" / "sweeps" / "demo" / "1714069323"
    aggregate.mkdir(parents=True)
    (aggregate / "aggregate.json").write_text("{}")

    path = sweep_aggregate_api_path(
        namespace="bench", sweep_name="demo", sweep_run_epoch="1714069323"
    )
    assert path.startswith(API_RESULTS_FILES_PATH)
    relative_path = path.removeprefix(API_RESULTS_FILES_PATH).lstrip("/")
    assert (tmp_path / relative_path).is_file()
