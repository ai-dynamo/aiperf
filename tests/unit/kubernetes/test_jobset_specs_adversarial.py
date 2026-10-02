# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for Kubernetes JobSet spec generation.

Focuses on:
- reserved AIPerf labels and controller annotations staying authoritative
- JobSet ownerReferences boundaries, service DNS names, and replicatedJob names
- container args/env list-shape safety for shell-like user input
- results sidecar port/probe contract used by artifact harvesting
- podTemplate metadata merge behavior and invalid resource-shape rejection

Out of scope: live Kubernetes API submission and kopf create-handler retries; see
``tests/unit/operator/test_sweep_handler_adversarial.py`` for handler-level
state-machine and patch-shape regressions.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.config.deployment import PodTemplateConfig
from aiperf.kubernetes.constants import AIPerfLabels, Containers
from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.kubernetes.jobset import AIPerfJobSetSpec, controller_dns_name
from aiperf.kubernetes.jobset_helpers import build_security_context
from aiperf.kubernetes.jobset_specs import AIPerfContainerSpec

# ============================================================
# Helpers
# ============================================================


def _jobset_spec(**overrides: Any) -> AIPerfJobSetSpec:
    """Build a real JobSet spec with realistic benchmark identifiers."""
    baseline: dict[str, Any] = {
        "name": "aiperf-bench-7f2a",
        "namespace": "aiperf-benchmarks",
        "job_id": "aiperf-bench-7f2a",
        "image": "nvcr.io/nvidia/aiperf:custom-test-tag",
        "worker_replicas": 2,
        "workers_per_pod": 2,
        "record_processors_per_pod": 1,
    }
    baseline.update(overrides)
    return AIPerfJobSetSpec(**baseline)


def _manifest(**overrides: Any) -> dict[str, Any]:
    """Render the JobSet manifest from a real Pydantic spec."""
    return _jobset_spec(**overrides).to_k8s_manifest()


def _replicated_job(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one rendered replicatedJob by JobSet-level name."""
    return next(
        job for job in manifest["spec"]["replicatedJobs"] if job["name"] == name
    )


def _pod_metadata(replicated_job: dict[str, Any]) -> dict[str, Any]:
    """Return the nested PodTemplate metadata for a replicatedJob."""
    return replicated_job["template"]["spec"]["template"]["metadata"]


def _pod_spec(replicated_job: dict[str, Any]) -> dict[str, Any]:
    """Return the nested PodSpec for a replicatedJob."""
    return replicated_job["template"]["spec"]["template"]["spec"]


def _container(replicated_job: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one rendered container by name from a replicatedJob."""
    return next(
        container
        for container in _pod_spec(replicated_job)["containers"]
        if container["name"] == name
    )


def _env_by_name(container: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index a container env list by variable name."""
    return {item["name"]: item for item in container.get("env", [])}


# ============================================================
# Reserved metadata and ownership boundaries
# ============================================================


class TestJobSetSpecReservedMetadata:
    """Reserved labels/annotations must beat user-provided podTemplate metadata."""

    def test_pod_labels_user_collision_preserves_aiperf_reserved_values(self) -> None:
        template = PodTemplateConfig(
            labels={
                AIPerfLabels.APP_KEY: "rogue-app",
                AIPerfLabels.JOB_ID: "rogue-benchmark",
                "team.nvidia.com/owner": "perf-infra",
            }
        )

        manifest = _manifest(pod_template=template)

        for replicated_job_name in ("controller", "workers"):
            labels = _pod_metadata(_replicated_job(manifest, replicated_job_name))[
                "labels"
            ]
            assert labels[AIPerfLabels.APP_KEY] == AIPerfLabels.APP_VALUE
            assert labels[AIPerfLabels.JOB_ID] == "aiperf-bench-7f2a"
            assert labels["team.nvidia.com/owner"] == "perf-infra"

    def test_controller_prometheus_annotations_override_template_collision(
        self,
    ) -> None:
        template = PodTemplateConfig(
            annotations={
                "prometheus.io/scrape": "false",
                "prometheus.io/port": "1",
                "prometheus.io/path": "/do-not-scrape",
                "team.nvidia.com/owner": "perf-infra",
            }
        )

        manifest = _manifest(pod_template=template)
        annotations = _pod_metadata(_replicated_job(manifest, "controller"))[
            "annotations"
        ]

        assert annotations["prometheus.io/scrape"] == "true"
        assert annotations["prometheus.io/port"] == str(
            K8sEnvironment.PORTS.API_SERVICE
        )
        assert annotations["prometheus.io/path"] == "/metrics"
        assert annotations["team.nvidia.com/owner"] == "perf-infra"

    def test_to_k8s_manifest_does_not_forge_owner_references(self) -> None:
        """OwnerReferences are injected by the kopf create handler with the live CR UID."""
        manifest = _manifest(
            extra_annotations={"aiperf.nvidia.com/requested-by": "ci-smoke-7f2a"}
        )

        assert "ownerReferences" not in manifest["metadata"]
        assert manifest["metadata"]["annotations"] == {
            "aiperf.nvidia.com/requested-by": "ci-smoke-7f2a"
        }


# ============================================================
# Names and JobSet topology
# ============================================================


class TestJobSetSpecNamesAndTopology:
    """Service names and replicatedJob names are a cross-container wire contract."""

    def test_replicated_job_names_match_success_policy_target(self) -> None:
        manifest = _manifest()

        replicated_job_names = [
            job["name"] for job in manifest["spec"]["replicatedJobs"]
        ]
        assert replicated_job_names == ["controller", "workers"]
        assert manifest["spec"]["successPolicy"] == {
            "operator": "All",
            "targetReplicatedJobs": ["controller"],
        }

    @pytest.mark.parametrize(
        "jobset_name,namespace,expected",
        [
            (
                "aiperf-bench-7f2a",
                "aiperf-benchmarks",
                "aiperf-bench-7f2a-controller-0-0."
                "aiperf-bench-7f2a.aiperf-benchmarks.svc.cluster.local",
            ),
            param(
                "latency-sweep-v03",
                "perf-canary",
                "latency-sweep-v03-controller-0-0."
                "latency-sweep-v03.perf-canary.svc.cluster.local",
                id="sweep-style-name",
            ),
        ],
    )  # fmt: skip
    def test_controller_dns_name_uses_jobset_headless_service_shape(
        self, jobset_name: str, namespace: str, expected: str
    ) -> None:
        assert controller_dns_name(jobset_name, namespace) == expected

    @pytest.mark.parametrize(
        "jobset_name",
        [
            param("evil.com:8080/", id="scheme-relative-host-port-path"),
            param("user:pw@evil.com", id="url-userinfo-credential"),
            param("http://evil.com", id="explicit-scheme"),
            param("../../etc/passwd", id="dot-segment-traversal"),
            param("bench..evil", id="empty-dot-label"),
            param(".leading-dot", id="leading-dot"),
            param("trailing-dot.", id="trailing-dot"),
            param("-leading-hyphen", id="leading-hyphen"),
            param("UPPERCASE", id="uppercase"),
            param("bench name", id="embedded-space"),
            param("bench\nhost: evil", id="embedded-newline"),
            param("bench#frag", id="fragment"),
            param("", id="empty"),
            param("a" * 254, id="over-subdomain-length"),
        ],
    )  # fmt: skip
    def test_controller_dns_name_rejects_non_rfc1123_jobset_name(
        self, jobset_name: str
    ) -> None:
        """`status.jobSetName` is writable by any principal holding `patch` on
        `aiperfjobs/status`, and the result is dialed directly over HTTP/WS.

        Anything that is not a DNS-1123 subdomain must raise rather than be
        interpolated into the operator's upstream URL, or a status write becomes
        SSRF.
        """
        with pytest.raises(ValueError, match="Invalid JobSet name"):
            controller_dns_name(jobset_name, "aiperf-benchmarks")

    @pytest.mark.parametrize(
        "namespace",
        [
            param("evil.com:8080/", id="host-port-path"),
            param("has.dots", id="dots-are-not-labels"),
            param("-leading-hyphen", id="leading-hyphen"),
            param("UPPER", id="uppercase"),
            param("", id="empty"),
            param("n" * 64, id="over-label-length"),
        ],
    )  # fmt: skip
    def test_controller_dns_name_rejects_non_rfc1123_namespace(
        self, namespace: str
    ) -> None:
        """Namespaces are DNS-1123 labels, so a dot is enough to re-root the host."""
        with pytest.raises(ValueError, match="Invalid namespace"):
            controller_dns_name("aiperf-bench-7f2a", namespace)

    @pytest.mark.parametrize(
        "jobset_name",
        [
            param("a", id="single-char"),
            param("bench-7f2a", id="hyphenated"),
            param("bench.sub.domain", id="multi-label-subdomain"),
            param("0-starts-with-digit", id="leading-digit"),
            param("a" * 253, id="at-subdomain-length-limit"),
        ],
    )  # fmt: skip
    def test_controller_dns_name_accepts_valid_kubernetes_names(
        self, jobset_name: str
    ) -> None:
        """The validation must not reject names Kubernetes itself accepts."""
        host = controller_dns_name(jobset_name, "aiperf-benchmarks")
        assert host == (
            f"{jobset_name}-controller-0-0.{jobset_name}"
            ".aiperf-benchmarks.svc.cluster.local"
        )

    def test_worker_controller_host_env_matches_jobset_dns_contract(self) -> None:
        manifest = _manifest(name="aiperf-llama3-8b", namespace="perf-canary")
        worker_manager = _container(
            _replicated_job(manifest, "workers"), Containers.WORKER_GROUP_MANAGER
        )

        env = _env_by_name(worker_manager)
        assert env["AIPERF_K8S_ZMQ_CONTROLLER_HOST"]["value"] == (
            "aiperf-llama3-8b-controller-0-0."
            "aiperf-llama3-8b.perf-canary.svc.cluster.local"
        )
        assert manifest["spec"]["network"] == {"enableDNSHostnames": True}


# ============================================================
# Container args/env trust boundaries
# ============================================================


class TestJobSetSpecContainerArgsAndEnv:
    """Container args/env are lists of data, not shell-rendered command strings."""

    def test_service_args_remain_tokenized_when_env_contains_shell_metacharacters(
        self,
    ) -> None:
        template = PodTemplateConfig(
            env=[
                {
                    "name": "AIPERF_OPERATOR_NOTE",
                    "value": "--type api; touch /tmp/aiperf-owned $(whoami)",
                }
            ]
        )

        manifest = _manifest(pod_template=template)
        worker = _container(_replicated_job(manifest, "workers"), "worker-0")

        assert worker["command"] == ["aiperf"]
        assert worker["args"] == [
            "service",
            "--type",
            "worker",
            "--benchmark-run",
            f"{K8sEnvironment.JOBSET.CONFIG_MOUNT_PATH}/run_config.json",
            "--health-port",
            str(K8sEnvironment.PORTS.WORKER_HEALTH + 1),
            "--service-id",
            "worker_$(AIPERF_POD_INDEX)_0",
        ]
        assert _env_by_name(worker)["AIPERF_OPERATOR_NOTE"]["value"] == (
            "--type api; touch /tmp/aiperf-owned $(whoami)"
        )

    def test_reserved_env_cannot_be_overridden_by_later_pod_template_entry(
        self,
    ) -> None:
        template = PodTemplateConfig(
            env=[
                {"name": "AIPERF_JOB_ID", "value": "rogue-benchmark"},
                {"name": "AIPERF_NAMESPACE", "value": "rogue-namespace"},
            ]
        )

        manifest = _manifest(pod_template=template)
        control_plane = _container(
            _replicated_job(manifest, "controller"), Containers.CONTROL_PLANE
        )
        env_entries = control_plane["env"]
        env_names = [entry["name"] for entry in env_entries]

        assert env_names.count("AIPERF_JOB_ID") == 1
        assert env_names.count("AIPERF_NAMESPACE") == 1
        env = _env_by_name(control_plane)
        assert env["AIPERF_JOB_ID"]["value"] == "aiperf-bench-7f2a"
        assert env["AIPERF_NAMESPACE"]["value"] == "aiperf-benchmarks"

    def test_pod_index_env_uses_field_ref_not_shell_substitution(self) -> None:
        manifest = _manifest()
        worker = _container(_replicated_job(manifest, "workers"), "worker-0")

        pod_index = _env_by_name(worker)["AIPERF_POD_INDEX"]
        assert pod_index == {
            "name": "AIPERF_POD_INDEX",
            "valueFrom": {
                "fieldRef": {
                    "fieldPath": "metadata.labels['jobset.sigs.k8s.io/job-index']",
                }
            },
        }


# ============================================================
# Results sidecar contract
# ============================================================


class TestJobSetSpecResultsSidecarContract:
    """Results sidecar service/port settings must stay in lock-step."""

    def test_results_sidecar_port_env_and_probes_share_single_port_constant(
        self,
    ) -> None:
        manifest = _manifest()
        sidecar = _container(
            _replicated_job(manifest, "controller"), Containers.RESULTS_SIDECAR
        )
        expected_port = K8sEnvironment.PORTS.RESULTS_SIDECAR

        assert sidecar["command"] == [
            "python",
            "-m",
            "aiperf.kubernetes.results_sidecar",
        ]
        assert sidecar["ports"] == [{"containerPort": expected_port, "name": "results"}]
        assert _env_by_name(sidecar)["AIPERF_RESULTS_SIDECAR_PORT"]["value"] == str(
            expected_port
        )
        assert sidecar["startupProbe"]["httpGet"] == {
            "path": "/healthz",
            "port": expected_port,
        }
        assert sidecar["livenessProbe"]["httpGet"]["port"] == expected_port
        assert sidecar["readinessProbe"]["httpGet"]["port"] == expected_port

    def test_results_sidecar_mounts_only_results_and_tmp_volumes(self) -> None:
        manifest = _manifest()
        sidecar = _container(
            _replicated_job(manifest, "controller"), Containers.RESULTS_SIDECAR
        )

        assert sidecar["volumeMounts"] == [
            {"name": "results", "mountPath": "/results", "readOnly": True},
            {"name": "tmp", "mountPath": "/tmp"},
        ]
        assert _env_by_name(sidecar)["AIPERF_RESULTS_DIR"]["value"] == "/results"


# ============================================================
# PodTemplate merge and invalid shapes
# ============================================================


class TestJobSetSpecPodTemplateMergeAndInvalidShapes:
    """PodTemplateConfig is a trust boundary for arbitrary K8s-native fragments."""

    def test_pod_template_extra_pod_spec_cannot_override_security_context(
        self,
    ) -> None:
        """extraPodSpec is an escape hatch, not a hardening bypass."""
        with pytest.raises(ValidationError, match="security-critical"):
            PodTemplateConfig(
                pod_security_context={"fsGroup": 2000},
                extra_pod_spec={
                    "securityContext": {"runAsNonRoot": False, "runAsUser": 0}
                },
            )

    @pytest.mark.parametrize(
        "key, value",
        [
            param("securityContext", {"runAsUser": 0}, id="securityContext"),
            param("hostNetwork", True, id="hostNetwork"),
            param("hostPID", True, id="hostPID"),
            param("hostIPC", True, id="hostIPC"),
            param("hostUsers", False, id="hostUsers"),
            param("containers", [{"name": "evil", "image": "evil"}], id="containers"),
        ],
    )  # fmt: skip
    def test_extra_pod_spec_denied_keys_rejected(self, key: str, value: Any) -> None:
        """Every security-critical PodSpec key is refused at validation time."""
        with pytest.raises(ValidationError, match=key):
            PodTemplateConfig(extra_pod_spec={key: value})

    def test_extra_pod_spec_denied_key_stripped_when_validation_bypassed(self) -> None:
        """Renderer drops denied keys even if a template dodges validation.

        ``model_construct`` skips validators, standing in for any construction
        path that does not round-trip through CRD validation.
        """
        template = PodTemplateConfig.model_construct(
            extra_pod_spec={
                "securityContext": {"runAsNonRoot": False, "runAsUser": 0},
                "hostPID": True,
                "schedulingGates": [{"name": "bench-ready"}],
            }
        )

        manifest = _manifest(pod_template=template)
        pod_spec = _pod_spec(_replicated_job(manifest, "controller"))

        assert pod_spec["securityContext"]["runAsNonRoot"] is True
        assert pod_spec["securityContext"]["runAsUser"] == 1000
        assert "hostPID" not in pod_spec
        # Non-security keys still flow through the escape hatch.
        assert pod_spec["schedulingGates"] == [{"name": "bench-ready"}]

    def test_extra_pod_spec_allows_unmodeled_keys(self) -> None:
        """The escape hatch keeps working for its intended purpose."""
        template = PodTemplateConfig(
            extra_pod_spec={"schedulingGates": [{"name": "bench-ready"}]}
        )

        manifest = _manifest(pod_template=template)
        pod_spec = _pod_spec(_replicated_job(manifest, "controller"))

        assert pod_spec["schedulingGates"] == [{"name": "bench-ready"}]

    def test_pod_security_context_benign_override_merges_over_hardened_base(
        self,
    ) -> None:
        """Non-escalating pod securityContext overrides still merge normally."""
        template = PodTemplateConfig(pod_security_context={"fsGroup": 2000})

        manifest = _manifest(pod_template=template)
        pod_spec = _pod_spec(_replicated_job(manifest, "controller"))

        assert pod_spec["securityContext"]["fsGroup"] == 2000
        assert pod_spec["securityContext"]["runAsNonRoot"] is True

    @pytest.mark.parametrize(
        "override",
        [
            param({"privileged": True}, id="privileged"),
            param({"allowPrivilegeEscalation": True}, id="allowPrivilegeEscalation"),
            param({"runAsNonRoot": False}, id="runAsNonRoot-false"),
            param({"runAsUser": 0}, id="runAsUser-root"),
            param({"runAsGroup": 0}, id="runAsGroup-root"),
        ],
    )  # fmt: skip
    def test_container_security_context_privilege_escalation_rejected(
        self, override: dict[str, Any]
    ) -> None:
        """container_security_context cannot weaken the hardened container context."""
        with pytest.raises(ValidationError, match="privilege-escalating"):
            PodTemplateConfig(container_security_context=override)

    def test_pod_security_context_privilege_escalation_rejected(self) -> None:
        """pod_security_context is gated identically to the container one."""
        with pytest.raises(ValidationError, match="privilege-escalating"):
            PodTemplateConfig(pod_security_context={"runAsNonRoot": False})

    def test_container_security_context_escalation_stripped_by_builder(self) -> None:
        """build_security_context drops escalating values if validation is bypassed."""
        template = PodTemplateConfig.model_construct(
            container_security_context={
                "privileged": True,
                "allowPrivilegeEscalation": True,
                "runAsUser": 0,
                "readOnlyRootFilesystem": False,
            }
        )

        ctx = build_security_context(template, job_identity="stripped-job")

        assert "privileged" not in ctx
        assert ctx["allowPrivilegeEscalation"] is False
        assert ctx["runAsUser"] == 1000
        # Non-escalating overrides still apply.
        assert ctx["readOnlyRootFilesystem"] is False

    def test_container_security_context_benign_overrides_still_apply(self) -> None:
        """Legitimate overrides are unaffected by the escalation gate."""
        template = PodTemplateConfig(
            container_security_context={
                "readOnlyRootFilesystem": False,
                "capabilities": {"add": ["NET_ADMIN"]},
            }
        )

        ctx = build_security_context(template, job_identity="benign-job")

        assert ctx["readOnlyRootFilesystem"] is False
        assert ctx["capabilities"] == {"drop": ["ALL"], "add": ["NET_ADMIN"]}
        assert ctx["runAsNonRoot"] is True

    @pytest.mark.parametrize(
        "resources",
        [
            param(["cpu=500m"], id="list-when-mapping"),
            param({"requests": ["cpu=500m"]}, id="nested-list-when-mapping"),
            param({"requests": {"cpu": ["500m"]}}, id="list-when-quantity-string"),
        ],
    )  # fmt: skip
    def test_container_resources_malformed_shape_raises_with_field_name(
        self, resources: object
    ) -> None:
        with pytest.raises(ValueError, match="resources"):
            AIPerfContainerSpec(
                name="worker-0",
                image="nvcr.io/nvidia/aiperf:custom-test-tag",
                resources=resources,
            )

    def test_pod_template_negative_termination_grace_rejected_with_field_name(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="termination_grace_period_seconds"):
            PodTemplateConfig(termination_grace_period_seconds=-1)
