# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Init-container rendering in the JobSet pod spec."""

from typing import Any

import pytest
from pytest import param

from aiperf.config.deployment import PodTemplateConfig
from aiperf.kubernetes.jobset_helpers import _warn_once, build_security_context
from aiperf.kubernetes.jobset_specs import AIPerfReplicatedJobSpec


def _pod_spec(template: PodTemplateConfig) -> dict[str, Any]:
    return AIPerfReplicatedJobSpec(
        name="bench", pod_template=template
    )._build_pod_spec()


class TestInitContainerRendering:
    """Bare init containers get the hardened default; explicit ones pass through."""

    def test_build_pod_spec_bare_init_container_gets_hardened_default(self) -> None:
        spec = _pod_spec(
            PodTemplateConfig(init_containers=[{"name": "a", "image": "busybox"}])
        )
        ctx = spec["initContainers"][0]["securityContext"]
        assert ctx["runAsNonRoot"] is True
        assert ctx["allowPrivilegeEscalation"] is False
        assert ctx["capabilities"] == {"drop": ["ALL"]}
        assert ctx["seccompProfile"] == {"type": "RuntimeDefault"}

    def test_build_pod_spec_bare_init_container_default_omits_read_only_root(
        self,
    ) -> None:
        """Init containers lack AIPerf's writable emptyDir layout."""
        spec = _pod_spec(
            PodTemplateConfig(init_containers=[{"name": "a", "image": "busybox"}])
        )
        assert (
            "readOnlyRootFilesystem" not in spec["initContainers"][0]["securityContext"]
        )

    def test_build_pod_spec_explicit_init_container_context_passes_through(
        self,
    ) -> None:
        """Privileged setup work must survive rendering untouched."""
        supplied = {"privileged": True, "runAsUser": 0}
        spec = _pod_spec(
            PodTemplateConfig(
                init_containers=[
                    {"name": "sysctl", "image": "busybox", "securityContext": supplied}
                ]
            )
        )
        assert spec["initContainers"][0]["securityContext"] == supplied

    def test_build_pod_spec_init_container_default_honors_container_overrides(
        self,
    ) -> None:
        spec = _pod_spec(
            PodTemplateConfig(
                container_security_context={"runAsUser": 65534},
                init_containers=[{"name": "a", "image": "busybox"}],
            )
        )
        assert spec["initContainers"][0]["securityContext"]["runAsUser"] == 65534

    def test_build_pod_spec_no_init_containers_omits_key(self) -> None:
        assert "initContainers" not in _pod_spec(PodTemplateConfig())

    def test_build_pod_spec_init_container_input_is_not_mutated(self) -> None:
        template = PodTemplateConfig(
            init_containers=[{"name": "a", "image": "busybox"}]
        )
        _pod_spec(template)
        assert template.init_containers[0] == {"name": "a", "image": "busybox"}

    def test_build_pod_spec_host_path_volume_renders_unchanged(self) -> None:
        """Warned-about volumes still reach the rendered PodSpec."""
        spec = AIPerfReplicatedJobSpec(
            name="bench",
            volumes=[{"name": "h", "hostPath": {"path": "/dev"}}],
        )._build_pod_spec()
        assert spec["volumes"] == [{"name": "h", "hostPath": {"path": "/dev"}}]


class TestSecurityContextMergeWarnings:
    """The merge path warns about widening overrides and still applies them."""

    def test_build_security_context_capability_add_merges_and_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        _warn_once.cache_clear()
        template = PodTemplateConfig(
            container_security_context={"capabilities": {"add": ["SYS_ADMIN"]}}
        )
        with caplog.at_level("WARNING"):
            ctx = build_security_context(template)
        assert ctx["capabilities"] == {"drop": ["ALL"], "add": ["SYS_ADMIN"]}
        assert any(
            rec.name == "aiperf.kubernetes.jobset_helpers"
            and "podTemplate.containerSecurityContext" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.parametrize(
        "overrides,key,expected",
        [
            param(
                {"readOnlyRootFilesystem": False},
                "readOnlyRootFilesystem",
                False,
                id="writable_root_filesystem",
            ),
            param(
                {"seccompProfile": {"type": "Unconfined"}},
                "seccompProfile",
                {"type": "Unconfined"},
                id="seccomp_unconfined",
            ),
        ],
    )  # fmt: skip
    def test_build_security_context_widening_override_passes_through(
        self, overrides: dict[str, Any], key: str, expected: Any
    ) -> None:
        _warn_once.cache_clear()
        ctx = build_security_context(
            PodTemplateConfig(container_security_context=overrides)
        )
        assert ctx[key] == expected

    def test_build_security_context_repeated_calls_warn_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """One warning per distinct message, not one per container rendered."""
        _warn_once.cache_clear()
        template = PodTemplateConfig(
            container_security_context={"seccompProfile": {"type": "Unconfined"}}
        )
        with caplog.at_level("WARNING"):
            for _ in range(5):
                build_security_context(template)
        matching = [
            rec
            for rec in caplog.records
            if rec.name == "aiperf.kubernetes.jobset_helpers"
            and "podTemplate.containerSecurityContext" in rec.message
        ]
        assert len(matching) == 1

    def test_build_security_context_escalating_key_still_dropped(self) -> None:
        """Warn-only applies to widening keys, not to privilege escalation."""
        _warn_once.cache_clear()
        template = PodTemplateConfig.model_construct(
            container_security_context={"privileged": True},
            init_containers=[],
        )
        ctx = build_security_context(template)
        assert "privileged" not in ctx
