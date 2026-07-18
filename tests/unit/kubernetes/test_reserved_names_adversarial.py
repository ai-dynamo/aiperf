# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for reserved Kubernetes names across config surfaces.

Focuses on:
- podTemplate labels/env/annotations that collide with AIPerf-owned names
- AIPerfSweep childMetadata labels/annotations used for child selector identity
- sweep-controller JobSet env var ownership for parent sweep identity
- generated CR metadata avoiding user-controlled operator annotations
- CR completion-claim annotations not being trusted when supplied by users

Out of scope: generic podTemplate passthrough, covered by
``tests/unit/operator/test_pod_template_adversarial.py`` and worker fan-out in
``tests/unit/operator/test_worker_manifest_adversarial.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.config import BenchmarkConfig, BenchmarkRun
from aiperf.config.sweep import SweepVariation
from aiperf.kubernetes.constants import AIPerfLabels, Annotations, Containers
from aiperf.operator.handlers.create import _build_deployment
from aiperf.operator.handlers.sweep import create as sweep_create
from aiperf.operator.models import AIPerfJobSpec, AIPerfSweepSpec
from aiperf.sweep_controller.k8s_executor import (
    SWEEP_LABEL,
    SWEEP_RUN_EPOCH_LABEL,
    SWEEP_UID_LABEL,
    TRIAL_INDEX_LABEL,
    VARIATION_INDEX_LABEL,
    VARIATION_LABEL_LABEL,
    VARIATION_VALUES_ANNOTATION,
    K8sChildJobExecutor,
)
from tests.harness.operator import build_minimal_aiperfjob_spec

# =============================================================================
# Helpers
# =============================================================================


# Cellular runner pods (controller + cell) share the HF tokenizer cache
# location and the CR-identity env AIPerf owns. These are the vars a hostile
# podTemplate must not be able to seize on any runner pod.
_RESERVED_SHARED_RUNTIME_ENV = frozenset(
    {
        "HF_HOME",
        "AIPERF_JOB_ID",
        "AIPERF_NAMESPACE",
    }
)

# The controller container additionally owns the cell-launcher selector that
# tells the runner the JobSet (not the runner) created the cell pods.
_RESERVED_CONTROLLER_RUNTIME_ENV = frozenset(
    {
        *_RESERVED_SHARED_RUNTIME_ENV,
        "AIPERF_CELL_LAUNCHER",
    }
)

# A cell container additionally owns its partition identity + the controller
# bootstrap coordinate + the HTTP artifact port.
_RESERVED_CELL_RUNTIME_ENV = frozenset(
    {
        *_RESERVED_SHARED_RUNTIME_ENV,
        "AIPERF_CELL_ID",
        "AIPERF_CELL_COUNT",
        "AIPERF_CELL_CONTROLLER_ADDR",
        "AIPERF_CELL_ARTIFACT_PORT",
    }
)


_RESERVED_SWEEP_CONTROLLER_ENV = frozenset(
    {
        "AIPERF_SWEEP_NAME",
        "AIPERF_SWEEP_NAMESPACE",
        "AIPERF_SWEEP_UID",
        "AIPERF_SWEEP_EPOCH",
        "AIPERF_RESULTS_DIR",
        "AIPERF_OPERATOR_BASE_URL",
    }
)


_VALID_BENCHMARK = {
    "models": ["meta-llama/Llama-3-8B"],
    "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
    "datasets": [{"name": "main", "type": "synthetic"}],
    "phases": [
        {
            "name": "profiling",
            "type": "concurrency",
            "requests": 10,
            "concurrency": 4,
        }
    ],
}


def _aiperfjob_spec_with(**overrides: Any) -> dict[str, Any]:
    """Build a real AIPerfJob spec by mutating the canonical validated baseline."""
    return {**build_minimal_aiperfjob_spec(), **overrides}


def _aiperfsweep_spec_with(**overrides: Any) -> dict[str, Any]:
    """Build a real AIPerfSweep spec with optional hostile metadata surfaces."""
    spec: dict[str, Any] = {
        "image": "nvcr.io/nvidia/aiperf:reserved-names",
        "sweep": {
            "type": "grid",
            "parameters": {"phases.profiling.concurrency": [4, 8]},
        },
        "multiRun": {"numRuns": 2},
        "benchmark": _VALID_BENCHMARK,
    }
    spec.update(overrides)
    return spec


def _manifest_from_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Render a JobSet manifest through AIPerfJobSpec and production conversion."""
    validated = AIPerfJobSpec.model_validate(spec)
    deployment, _ = _build_deployment(
        spec=validated.model_dump(by_alias=True, exclude_none=True),
        name="reserved-names-bench-7f2a",
        namespace="production-benchmarks",
        job_id="reserved-names-bench-7f2a",
    )
    return deployment.get_jobset_spec().to_k8s_manifest()


def _replicated_job(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a named JobSet replicatedJob fragment."""
    for replicated_job in manifest["spec"]["replicatedJobs"]:
        if replicated_job["name"] == name:
            return replicated_job
    raise AssertionError(f"replicatedJob {name!r} not found")


def _pod_spec(manifest: dict[str, Any], replicated_job_name: str) -> dict[str, Any]:
    """Return the rendered pod spec for a replicatedJob."""
    return _replicated_job(manifest, replicated_job_name)["template"]["spec"][
        "template"
    ]["spec"]


def _pod_metadata(manifest: dict[str, Any], replicated_job_name: str) -> dict[str, Any]:
    """Return the rendered pod-template metadata for a replicatedJob."""
    return _replicated_job(manifest, replicated_job_name)["template"]["spec"][
        "template"
    ]["metadata"]


def _container(
    manifest: dict[str, Any], replicated_job_name: str, container_name: str
) -> dict[str, Any]:
    """Return a named container from a rendered replicatedJob."""
    for container in _pod_spec(manifest, replicated_job_name)["containers"]:
        if container["name"] == container_name:
            return container
    raise AssertionError(
        f"container {container_name!r} not found in {replicated_job_name!r}"
    )


def _env_entries(container: dict[str, Any], name: str) -> list[dict[str, Any]]:
    """Return all env entries with ``name`` so duplicate reserved keys are visible."""
    return [entry for entry in container.get("env", []) if entry.get("name") == name]


def _env_by_name(container: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index Kubernetes EnvVar entries by name."""
    return {entry["name"]: entry for entry in container.get("env", [])}


def _benchmark_run() -> BenchmarkRun:
    """Build a realistic child run with variation metadata for sweep children."""
    benchmark = BenchmarkConfig.model_validate(_VALID_BENCHMARK)
    return BenchmarkRun(
        benchmark_id="reserved-sweep-v03-t7",
        cfg=benchmark,
        variation=SweepVariation(
            index=3,
            label="Concurrency / 4 + TTFT SLA",
            values={"benchmark.phases.profiling.concurrency": 4},
        ),
        trial=7,
        artifact_dir=Path("/results/production-benchmarks/reserved-sweep-v03-t7"),
        label="concurrency_4_trial_7",
        cli_command=None,
    )


@asynccontextmanager
async def _fake_k8s_client() -> AsyncIterator[MagicMock]:
    """Yield a mock ApiClient without opening a real Kubernetes connection."""
    yield MagicMock(name="ApiClient")


async def _capture_sweep_controller_jobset(
    monkeypatch: pytest.MonkeyPatch,
    *,
    template_spec: dict[str, Any],
) -> dict[str, Any]:
    """Create the sweep-controller JobSet and capture the submitted manifest body."""
    captured: dict[str, Any] = {}

    async def _capture_create(**kwargs: Any) -> None:
        captured["body"] = kwargs["body"]

    custom = MagicMock(
        create_namespaced_custom_object=AsyncMock(side_effect=_capture_create)
    )
    monkeypatch.setattr(
        "aiperf.kubernetes.client.k8s_client",
        lambda **_kwargs: _fake_k8s_client(),
        raising=True,
    )
    monkeypatch.setattr(
        "kubernetes_asyncio.client.CustomObjectsApi", lambda _api: custom
    )

    await sweep_create._create_sweep_controller_jobset(
        name="reserved-sweep",
        namespace="production-benchmarks",
        sweep_uid="uid-reserved-sweep-7f2a",
        epoch="1778027130",
        template_spec=template_spec,
    )
    return captured["body"]


# =============================================================================
# podTemplate trust boundary
# =============================================================================


class TestPodTemplateReservedNames:
    """podTemplate cannot seize labels/env/annotations owned by AIPerf."""

    def test_jobset_manifest_pod_template_reserved_labels_keep_aiperf_values(
        self,
    ) -> None:
        spec = _aiperfjob_spec_with(
            podTemplate={
                "labels": {
                    "app": "not-aiperf",
                    AIPerfLabels.JOB_ID: "hijacked-job-id",
                    "team.nvidia.com/owner": "platform-benchmarks",
                }
            }
        )

        manifest = _manifest_from_spec(spec)

        # Cellular JobSet emits `controller` + `cells` replicatedJobs (the mesh
        # `workers` job is retired). Reserved app/job-id labels stay AIPerf-owned
        # on both; a user's own label survives.
        for replicated_job_name in ("controller", "cells"):
            labels = _pod_metadata(manifest, replicated_job_name)["labels"]
            assert labels[AIPerfLabels.APP_KEY] == AIPerfLabels.APP_VALUE
            assert labels[AIPerfLabels.JOB_ID] == "reserved-names-bench-7f2a"
            assert labels["team.nvidia.com/owner"] == "platform-benchmarks"

    @pytest.mark.parametrize(
        "replicated_job_name,container_name,reserved_names",
        [
            param(
                "controller",
                Containers.CELL_CONTROLLER,
                _RESERVED_CONTROLLER_RUNTIME_ENV,
                id="controller",
            ),
            param(
                "cells",
                "cell",
                _RESERVED_CELL_RUNTIME_ENV,
                id="cell",
            ),
        ],
    )  # fmt: skip
    def test_jobset_manifest_pod_template_reserved_env_stays_authoritative(
        self,
        replicated_job_name: str,
        container_name: str,
        reserved_names: frozenset[str],
    ) -> None:
        """A hostile podTemplate cannot make a reserved env var resolve to its value.

        The cellular runner layers AIPerf-owned env (HF cache, CR identity, cell
        partition) *after* the user's podTemplate env, so for a name the user also
        set there are two entries. Kubernetes resolves duplicate env names
        last-wins, so the effective (last) entry is always AIPerf's — never the
        attacker's. HF_HOME is deduped outright by build_runner_env_vars.
        """
        hostile_env = [
            {"name": name, "value": "attacker-controlled"}
            for name in sorted(reserved_names)
        ]
        spec = _aiperfjob_spec_with(podTemplate={"env": hostile_env})

        manifest = _manifest_from_spec(spec)
        container = _container(manifest, replicated_job_name, container_name)

        for name in reserved_names:
            entries = _env_entries(container, name)
            assert entries, f"reserved env {name!r} missing from {container_name!r}"
            # k8s last-wins: the effective value is the final entry for this name.
            effective = entries[-1]
            assert effective.get("value") != "attacker-controlled", (
                f"reserved env {name!r} resolved to the attacker value"
            )

    def test_jobset_manifest_pod_template_annotations_pass_through_on_runner_pods(
        self,
    ) -> None:
        """The cellular runner pods carry no AIPerf-owned pod annotations.

        The retired mesh controller stamped a Prometheus scrape contract onto its
        pod annotations, so user annotations were reconciled against reserved
        keys. Cellular runner pods (controller + cells) own no pod annotations, so
        user-supplied podTemplate annotations pass through verbatim on every job.
        """
        spec = _aiperfjob_spec_with(
            podTemplate={
                "annotations": {
                    "prometheus.io/scrape": "false",
                    "prometheus.io/port": "1",
                    "observability.nvidia.com/runbook": "https://runbooks.nvidia.com/aiperf",
                }
            }
        )

        manifest = _manifest_from_spec(spec)

        for replicated_job_name in ("controller", "cells"):
            annotations = _pod_metadata(manifest, replicated_job_name)["annotations"]
            assert annotations["prometheus.io/scrape"] == "false"
            assert annotations["prometheus.io/port"] == "1"
            assert annotations["observability.nvidia.com/runbook"].endswith("/aiperf")


# =============================================================================
# AIPerfSweep child metadata trust boundary
# =============================================================================


class TestChildMetadataReservedNames:
    """childMetadata keeps sweep-owned selector labels and variation annotations."""

    def test_child_metadata_reserved_selector_labels_override_user_values(self) -> None:
        spec = AIPerfSweepSpec.model_validate(
            _aiperfsweep_spec_with(
                childMetadata={
                    "labels": {
                        SWEEP_LABEL: "attacker-sweep",
                        SWEEP_UID_LABEL: "uid-attacker",
                        SWEEP_RUN_EPOCH_LABEL: "9999999999",
                        VARIATION_INDEX_LABEL: "99",
                        VARIATION_LABEL_LABEL: "wrong-label",
                        TRIAL_INDEX_LABEL: "9",
                        "team.nvidia.com/owner": "perf-lab",
                    },
                    "annotations": {
                        VARIATION_VALUES_ANNOTATION: '{"poisoned":true}',
                        "runbook.nvidia.com/url": "https://runbooks.nvidia.com/aiperf-sweep",
                    },
                }
            )
        )
        executor = K8sChildJobExecutor(
            api=None,
            sweep={
                "metadata": {
                    "name": "reserved-sweep",
                    "namespace": "production-benchmarks",
                    "uid": "uid-reserved-sweep-7f2a",
                },
                "spec": spec.model_dump(by_alias=True, exclude_none=True),
            },
            with_trial_suffix=True,
            sweep_run_epoch="1778027130",
        )

        metadata = executor._build_child_metadata(
            _benchmark_run(), "reserved-sweep-v03-t7"
        )
        labels = metadata["labels"]
        annotations = metadata["annotations"]

        assert labels[SWEEP_LABEL] == "reserved-sweep"
        assert labels[SWEEP_UID_LABEL] == "uid-reserved-sweep-7f2a"
        assert labels[SWEEP_RUN_EPOCH_LABEL] == "1778027130"
        assert labels[VARIATION_INDEX_LABEL] == "03"
        assert labels[VARIATION_LABEL_LABEL] == "concurrency-4-ttft-sla"
        assert labels[TRIAL_INDEX_LABEL] == "7"
        assert labels["team.nvidia.com/owner"] == "perf-lab"
        assert annotations[VARIATION_VALUES_ANNOTATION] == (
            '{"benchmark.phases.profiling.concurrency":4}'
        )
        assert annotations["runbook.nvidia.com/url"].endswith("/aiperf-sweep")

    @pytest.mark.parametrize(
        "forbidden_key",
        [
            "name",
            "namespace",
            "uid",
            param("ownerReferences", id="owner-references"),
        ],
    )  # fmt: skip
    def test_child_metadata_objectmeta_identity_fields_rejected(
        self, forbidden_key: str
    ) -> None:
        with pytest.raises(
            ValidationError, match=rf"(?i)childMetadata|{forbidden_key}|extra"
        ):
            AIPerfSweepSpec.model_validate(
                _aiperfsweep_spec_with(
                    childMetadata={forbidden_key: "attacker-controlled"}
                )
            )


# =============================================================================
# Sweep-controller generated manifest trust boundary
# =============================================================================


class TestSweepControllerReservedNames:
    """The generated sweep-controller JobSet owns parent-identity env vars."""

    @pytest.mark.asyncio
    async def test_sweep_controller_jobset_pod_template_reserved_env_cannot_override_identity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        hostile_env = [
            {"name": name, "value": "attacker-controlled"}
            for name in sorted(_RESERVED_SWEEP_CONTROLLER_ENV)
        ]
        hostile_env.append(
            {"name": "HTTP_PROXY", "value": "http://proxy.internal:8080"}
        )

        body = await _capture_sweep_controller_jobset(
            monkeypatch,
            template_spec={
                "image": "nvcr.io/nvidia/aiperf:sweep-controller",
                "podTemplate": {"env": hostile_env},
            },
        )
        pod_spec = body["spec"]["replicatedJobs"][0]["template"]["spec"]["template"][
            "spec"
        ]
        containers = {
            container["name"]: container for container in pod_spec["containers"]
        }
        env = _env_by_name(containers["sweep-controller"])
        sidecar_env = _env_by_name(containers[Containers.RESULTS_SIDECAR])

        for name in _RESERVED_SWEEP_CONTROLLER_ENV:
            assert len(_env_entries(containers["sweep-controller"], name)) == 1
            assert env[name].get("value") != "attacker-controlled"
        assert env["AIPERF_SWEEP_NAME"]["value"] == "reserved-sweep"
        assert env["AIPERF_SWEEP_UID"]["value"] == "uid-reserved-sweep-7f2a"
        assert env["AIPERF_SWEEP_EPOCH"]["value"] == "1778027130"
        assert env["HTTP_PROXY"]["value"] == "http://proxy.internal:8080"
        assert sidecar_env["AIPERF_RESULTS_DIR"]["value"] == "/results"


# =============================================================================
# CR annotation trust boundary
# =============================================================================


class TestCrAnnotationReservedNames:
    """User-controlled CR annotations must not impersonate operator-owned claims."""

    @pytest.mark.asyncio
    async def test_try_claim_completion_user_supplied_reserved_annotation_does_not_skip_claim(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator import client_cache

        client_cache._reset_for_testing()
        submit_claim = AsyncMock(return_value=True)
        monkeypatch.setattr(client_cache, "_submit_claim_patch", submit_claim)
        monkeypatch.setattr(client_cache, "_post_dashboard_refresh", AsyncMock())
        body = {
            "metadata": {
                "name": "reserved-names-bench-7f2a",
                "namespace": "production-benchmarks",
                "annotations": {
                    Annotations.COMPLETION_CLAIMED: "attacker-controlled",
                    "owner.nvidia.com/team": "perf-lab",
                },
            }
        }

        claimed = await client_cache.try_claim_completion(
            "production-benchmarks", "reserved-names-bench-7f2a", body
        )

        assert claimed is True
        submit_claim.assert_awaited_once()
