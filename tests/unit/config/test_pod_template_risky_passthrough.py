# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate-and-warn behavior for the free-form podTemplate passthroughs.

AIPerf is a benchmarking tool: hostPath volumes and privileged init containers
are legitimate and must never be rejected. These tests pin the warn-only
contract so a future change cannot quietly turn it into a denial.
"""

from typing import Any

import pytest
from pytest import param

from aiperf.config.deployment import (
    PodTemplateConfig,
    risky_init_container_warnings,
    risky_security_context_details,
    risky_security_context_warnings,
    risky_volume_warnings,
)


class TestRiskyVolumeWarnings:
    """`risky_volume_warnings` describes host-backed volume sources."""

    @pytest.mark.parametrize(
        "volume,expected_fragment",
        [
            param(
                {"name": "h", "hostPath": {"path": "/mnt/nvme"}},
                "hostPath volume mounting node path '/mnt/nvme'",
                id="host_path_reports_path",
            ),
            param(
                {"name": "n", "nfs": {"server": "s", "path": "/e"}},
                "nfs volume backed by storage outside the pod",
                id="nfs_reported",
            ),
            param(
                {"name": "i", "iscsi": {"targetPortal": "t"}},
                "iscsi volume backed by storage outside the pod",
                id="iscsi_reported",
            ),
        ],
    )  # fmt: skip
    def test_risky_volume_warnings_risky_source_returns_warning(
        self, volume: dict[str, Any], expected_fragment: str
    ) -> None:
        warnings = risky_volume_warnings([volume])
        assert len(warnings) == 1
        assert expected_fragment in warnings[0]
        assert "podTemplate.volumes[0]" in warnings[0]

    @pytest.mark.parametrize(
        "path,sensitive",
        [
            param("/", True, id="node_root"),
            param("/etc/kubernetes", True, id="etc_subdir"),
            param("/var/run/docker.sock", True, id="runtime_socket"),
            param("/var/lib/kubelet", True, id="kubelet_state_exact"),
            param("/dev/shm", False, id="dev_shm_benign"),
            param("/mnt/nvme0", False, id="scratch_benign"),
            param("/etcd-data", False, id="prefix_lookalike_benign"),
        ],
    )  # fmt: skip
    def test_risky_volume_warnings_host_path_sensitivity_flagged_only_when_sensitive(
        self, path: str, sensitive: bool
    ) -> None:
        warning = risky_volume_warnings([{"name": "v", "hostPath": {"path": path}}])[0]
        assert ("exposes node credentials" in warning) is sensitive

    def test_risky_volume_warnings_safe_volumes_returns_empty(self) -> None:
        volumes = [
            {"name": "e", "emptyDir": {}},
            {"name": "c", "configMap": {"name": "cfg"}},
            {"name": "p", "persistentVolumeClaim": {"claimName": "pvc"}},
        ]
        assert risky_volume_warnings(volumes) == []

    def test_risky_volume_warnings_non_dict_entry_is_skipped(self) -> None:
        assert risky_volume_warnings(["not-a-volume"]) == []  # type: ignore[list-item]


class TestRiskyInitContainerWarnings:
    """`risky_init_container_warnings` describes unhardened init containers."""

    def test_risky_init_container_warnings_no_security_context_returns_empty(
        self,
    ) -> None:
        """A bare init container gets the hardened default, so nothing to warn about."""
        assert risky_init_container_warnings([{"name": "a", "image": "busybox"}]) == []

    def test_risky_init_container_warnings_privileged_names_the_key(self) -> None:
        warnings = risky_init_container_warnings(
            [{"name": "sysctl", "securityContext": {"privileged": True}}]
        )
        assert len(warnings) == 1
        assert "podTemplate.initContainers[0] ('sysctl')" in warnings[0]
        assert "privileged=True" in warnings[0]

    def test_risky_init_container_warnings_added_capabilities_are_listed(self) -> None:
        warnings = risky_init_container_warnings(
            [
                {
                    "name": "tune",
                    "securityContext": {"capabilities": {"add": ["SYS_ADMIN"]}},
                }
            ]
        )
        assert "adds Linux capabilities ['SYS_ADMIN']" in warnings[0]

    def test_risky_init_container_warnings_benign_context_still_notes_opt_out(
        self,
    ) -> None:
        warnings = risky_init_container_warnings(
            [{"name": "fix", "securityContext": {"runAsUser": 1234}}]
        )
        assert "hardened container baseline is not applied" in warnings[0]


class TestPodTemplateWarnOnlyContract:
    """Risky constructs must validate successfully and only log."""

    def test_pod_template_config_host_path_volume_is_accepted_and_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            template = PodTemplateConfig(
                volumes=[{"name": "gpu", "hostPath": {"path": "/dev"}}]
            )
        assert template.volumes[0]["hostPath"]["path"] == "/dev"
        assert any("podTemplate.volumes[0]" in rec.message for rec in caplog.records)

    def test_pod_template_config_privileged_init_container_is_accepted_and_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            template = PodTemplateConfig(
                init_containers=[
                    {
                        "name": "sysctl",
                        "image": "busybox",
                        "securityContext": {"privileged": True},
                    }
                ]
            )
        assert template.init_containers[0]["securityContext"]["privileged"] is True
        assert any(
            "podTemplate.initContainers[0]" in rec.message for rec in caplog.records
        )

    def test_pod_template_config_safe_template_logs_nothing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            PodTemplateConfig(
                volumes=[{"name": "e", "emptyDir": {}}],
                init_containers=[{"name": "a", "image": "busybox"}],
            )
        assert [rec for rec in caplog.records if "podTemplate." in rec.message] == []

    def test_pod_template_config_service_account_description_documents_token_mount(
        self,
    ) -> None:
        description = PodTemplateConfig.model_fields["service_account_name"].description
        assert description is not None
        assert "token" in description


class TestRiskySecurityContextWarnings:
    """Baseline-widening securityContext keys warn instead of being rejected."""

    @pytest.mark.parametrize(
        "ctx,expected_fragment",
        [
            param(
                {"capabilities": {"add": ["SYS_ADMIN"]}},
                "adds Linux capabilities ['SYS_ADMIN']",
                id="capabilities_add",
            ),
            param(
                {"readOnlyRootFilesystem": False},
                "disables readOnlyRootFilesystem",
                id="writable_root_filesystem",
            ),
            param(
                {"seccompProfile": {"type": "Unconfined"}},
                "sets seccompProfile.type='Unconfined'",
                id="seccomp_unconfined",
            ),
            param(
                {"capabilities": {"drop": ["NET_RAW"]}},
                "narrows the dropped capability set to ['NET_RAW'] instead of ALL",
                id="partial_capability_drop",
            ),
        ],
    )  # fmt: skip
    def test_risky_security_context_details_widening_key_returns_detail(
        self, ctx: dict[str, Any], expected_fragment: str
    ) -> None:
        assert expected_fragment in "; ".join(risky_security_context_details(ctx))

    @pytest.mark.parametrize(
        "ctx",
        [
            param({}, id="empty"),
            param({"runAsUser": 65534}, id="non_root_uid"),
            param({"capabilities": {"drop": ["ALL"]}}, id="drop_all"),
            param(
                {"seccompProfile": {"type": "RuntimeDefault"}},
                id="runtime_default_seccomp",
            ),
        ],
    )  # fmt: skip
    def test_risky_security_context_details_safe_context_returns_empty(
        self, ctx: dict[str, Any]
    ) -> None:
        assert risky_security_context_details(ctx) == []

    def test_risky_security_context_warnings_includes_field_path(self) -> None:
        warnings = risky_security_context_warnings(
            {"capabilities": {"add": ["SYS_PTRACE"]}},
            "podTemplate.containerSecurityContext",
        )
        assert len(warnings) == 1
        assert warnings[0].startswith("podTemplate.containerSecurityContext:")

    @pytest.mark.parametrize(
        "ctx",
        [
            param({"capabilities": {"add": ["SYS_ADMIN"]}}, id="capabilities_add"),
            param({"readOnlyRootFilesystem": False}, id="writable_root_filesystem"),
            param({"seccompProfile": {"type": "Unconfined"}}, id="seccomp_unconfined"),
        ],
    )  # fmt: skip
    def test_pod_template_config_widening_context_is_accepted_and_logged(
        self, ctx: dict[str, Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        """Profiling and GPU workloads need these; they must never be rejected."""
        with caplog.at_level("WARNING"):
            template = PodTemplateConfig(container_security_context=ctx)
        assert template.container_security_context == ctx
        assert any(
            "podTemplate.containerSecurityContext" in rec.message
            for rec in caplog.records
        )
