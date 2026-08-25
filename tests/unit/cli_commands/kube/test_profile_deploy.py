# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for `aiperf kube profile` deploy helpers.

`test_profile.py` already covers the top-level `profile()` command's
`--skip-endpoint-check` wiring. This file targets the helpers that
`profile` delegates to:
    - `profile._try_load_aiperfjob_cr`   — CR-vs-plain-config detection
    - `profile.generate_benchmark_name`  — deterministic DNS-safe name
    - `profile_deploy._build_cr`         — CR envelope construction
    - `profile_deploy.operator_available`— CRD probe (404 -> direct; 403/5xx -> raise)
    - `profile_deploy.wait_or_detach`    — interactive/detach split
    - `profile_deploy_direct._apply_manifest` — kind-dispatch table
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import param

from aiperf.cli_commands.kube.profile import (
    _try_load_aiperfjob_cr,
    generate_benchmark_name,
    profile,
)
from aiperf.cli_commands.kube.profile_deploy import (
    _build_cr,
    _replace_existing_cr_if_complete,
    _submit_cr,
    deploy_via_operator,
    operator_available,
    wait_or_detach,
)
from aiperf.cli_commands.kube.profile_deploy_direct import _apply_manifest
from aiperf.config.kube import KubeOptions
from aiperf.kubernetes.cr_refs import AIPERF_API_VERSION

# =============================================================================
# _try_load_aiperfjob_cr
# =============================================================================


class TestTryLoadAiperfjobCr:
    """Tests for the CR detection heuristic."""

    def test_valid_aiperfjob_cr_returns_dict(self, tmp_path) -> None:
        """A well-formed AIPerfJob YAML file is recognised."""
        cr_file = tmp_path / "job.yaml"
        cr_file.write_text(
            "apiVersion: aiperf.nvidia.com/v1alpha1\nkind: AIPerfJob\nspec: {}\n"
        )
        raw = _try_load_aiperfjob_cr(cr_file)
        assert raw is not None
        assert raw["kind"] == "AIPerfJob"

    @pytest.mark.parametrize(
        "content",
        [
            param("not-valid: yaml: [[[", id="malformed-yaml"),
            param("kind: Pod\napiVersion: v1\n", id="wrong-kind"),
            param("kind: AIPerfJob\napiVersion: other.io/v1\n", id="wrong-api-version"),
            param("just-a-string", id="not-a-mapping"),
            param("", id="empty"),
        ],
    )  # fmt: skip
    def test_non_cr_returns_none(self, tmp_path, content: str) -> None:
        """Non-AIPerfJob / malformed YAML paths return None."""
        cr_file = tmp_path / "other.yaml"
        cr_file.write_text(content)
        assert _try_load_aiperfjob_cr(cr_file) is None


# =============================================================================
# generate_benchmark_name
# =============================================================================


class TestGenerateBenchmarkName:
    """Tests for the benchmark-name generator."""

    def _stub_config(
        self,
        *,
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        endpoint_type: str = "chat",
        phase_type: str = "throughput",
    ) -> Any:
        """Build a stub config with the three fields the helper reads."""
        phase = MagicMock()
        phase.type = phase_type
        config = MagicMock()
        config.benchmark.get_model_names.return_value = [model]
        config.benchmark.endpoint.type = endpoint_type
        # phases is now list[PhaseConfig]; first-phase access is config.benchmark.phases[0]
        config.benchmark.phases = [phase]
        return config

    def test_basic_name(self) -> None:
        """Assembles model + endpoint-type + phase-type into a DNS-safe name."""
        config = self._stub_config()
        name = generate_benchmark_name(config)
        # Dots in the model tag are replaced with hyphens by the sanitizer
        assert name == "llama-3-1-8b-instruct-chat-throughput"
        # DNS-label shape: lower, hyphen-only, <=40
        assert len(name) <= 40
        assert all(c.islower() or c.isdigit() or c == "-" for c in name)

    def test_truncates_to_40(self) -> None:
        """Very long model names are truncated at 40 chars."""
        config = self._stub_config(model="x" * 100)
        assert len(generate_benchmark_name(config)) <= 40

    def test_strips_leading_and_trailing_hyphens(self) -> None:
        """Leading/trailing invalid chars are sanitized to hyphens then stripped."""
        config = self._stub_config(model="--weird--")
        out = generate_benchmark_name(config)
        assert not out.startswith("-")
        assert not out.endswith("-")


# =============================================================================
# _build_cr
# =============================================================================


class TestBuildCr:
    """Tests for the CR envelope builder."""

    def test_build_cr_shape(self) -> None:
        """`_build_cr` wraps spec with correct apiVersion/kind/metadata."""
        cr = _build_cr("my-bench", "ns-1", {"benchmark": {"foo": "bar"}})
        assert cr["apiVersion"] == AIPERF_API_VERSION
        assert cr["kind"] == "AIPerfJob"
        assert cr["metadata"] == {"name": "my-bench", "namespace": "ns-1"}
        assert cr["spec"] == {"benchmark": {"foo": "bar"}}


class TestSubmitCr:
    """The create response carries server-defaulted fields used in summaries."""

    async def test_returns_server_defaulted_cr(self) -> None:
        core = MagicMock()
        core.create_namespace = AsyncMock()
        custom = MagicMock()
        custom.create_namespaced_custom_object = AsyncMock(
            return_value={"spec": {"image": "registry.example/aiperf:chart-default"}}
        )
        cr = _build_cr("my-bench", "ns-1", {"benchmark": {"foo": "bar"}})

        with patch(
            "aiperf.cli_commands.kube.profile_deploy._replace_existing_cr_if_complete",
            new=AsyncMock(),
        ):
            created = await _submit_cr(
                custom,
                core,
                cr,
                name="my-bench",
                namespace="ns-1",
                kube_context=None,
                create_cli_owned_namespace=True,
            )

        assert created["spec"]["image"] == "registry.example/aiperf:chart-default"

    async def test_created_namespace_carries_job_specific_ownership(self) -> None:
        core = MagicMock()
        core.create_namespace = AsyncMock()
        custom = MagicMock()
        custom.create_namespaced_custom_object = AsyncMock(return_value={})
        cr = _build_cr("my-bench", "aiperf-my-bench", {"benchmark": {}})

        with patch(
            "aiperf.cli_commands.kube.profile_deploy._replace_existing_cr_if_complete",
            new=AsyncMock(),
        ):
            await _submit_cr(
                custom,
                core,
                cr,
                name="my-bench",
                namespace="aiperf-my-bench",
                kube_context=None,
                create_cli_owned_namespace=True,
            )

        namespace = core.create_namespace.await_args.kwargs["body"]
        assert namespace.metadata.labels == {
            "app": "aiperf",
            "aiperf.nvidia.com/auto-generated": "true",
            "aiperf.nvidia.com/job-id": "my-bench",
        }


class _ExistingCrApi:
    """Namespaced custom-object fake for replacement decisions."""

    def __init__(self, existing: dict[str, Any]) -> None:
        self.existing = existing
        self.deleted: list[str] = []

    async def get_namespaced_custom_object(self, **_: Any) -> dict[str, Any]:
        return self.existing

    async def delete_namespaced_custom_object(self, *, name: str, **_: Any) -> None:
        self.deleted.append(name)


class TestReplaceExistingCr:
    """A name collision may replace only a CR known to be terminal."""

    @pytest.mark.parametrize(
        "existing",
        [
            param({"status": {"phase": "Queued"}}, id="queued"),
            param({"status": {"phase": "Initializing"}}, id="initializing"),
            param({}, id="status-not-reported"),
            param({"status": {"phase": "FuturePhase"}}, id="unknown-phase"),
        ],
    )  # fmt: skip
    async def test_non_terminal_or_statusless_cr_is_preserved(
        self, existing: dict[str, Any]
    ) -> None:
        custom = _ExistingCrApi(existing)

        with pytest.raises(SystemExit, match="AIPerfJob my-bench"):
            await _replace_existing_cr_if_complete(
                custom,
                name="my-bench",
                namespace="tenant-a",
                kube_context=None,
            )

        assert custom.deleted == []

    @pytest.mark.parametrize(
        "phase",
        [
            param("Completed", id="completed"),
            param("Failed", id="failed"),
            param("Cancelled", id="cancelled"),
        ],
    )  # fmt: skip
    async def test_terminal_cr_is_replaced(self, phase: str) -> None:
        custom = _ExistingCrApi({"status": {"phase": phase}})

        await _replace_existing_cr_if_complete(
            custom,
            name="my-bench",
            namespace="tenant-a",
            kube_context=None,
        )

        assert custom.deleted == ["my-bench"]


class _RecordingCoreApi:
    """Core API fake that records namespace creation attempts."""

    def __init__(self) -> None:
        self.created_namespaces: list[str] = []

    async def create_namespace(self, *, body: Any) -> None:
        self.created_namespaces.append(body.metadata.name)


class _RecordingCustomApi:
    """Custom API fake that records submitted AIPerfJobs."""

    def __init__(self) -> None:
        self.created_jobs: list[tuple[str, str]] = []

    async def get_namespaced_custom_object(self, **_: Any) -> dict[str, Any]:
        from kubernetes_asyncio.client.exceptions import ApiException

        raise ApiException(status=404, reason="NotFound")

    async def create_namespaced_custom_object(
        self, *, namespace: str, body: dict[str, Any], **_: Any
    ) -> dict[str, Any]:
        self.created_jobs.append((namespace, body["metadata"]["name"]))
        return body


class TestDeployViaOperatorNamespaceOwnership:
    """Operator submission must not adopt explicitly managed namespaces."""

    async def _deploy(
        self, *, options: KubeOptions, namespace: str
    ) -> tuple[_RecordingCoreApi, _RecordingCustomApi]:
        from contextlib import asynccontextmanager

        core = _RecordingCoreApi()
        custom = _RecordingCustomApi()

        @asynccontextmanager
        async def _fake_client(**_: Any):
            yield object()

        validated = MagicMock()
        validated.image = "registry.example/aiperf:latest"
        validated.model_dump.return_value = {"image": validated.image}
        config = MagicMock()
        config.benchmark.endpoint.urls = []
        config.benchmark.get_model_names.return_value = ["model-a"]

        with (
            patch(
                "aiperf.kubernetes.spec_converter.validate_job_spec",
                return_value=validated,
            ),
            patch("aiperf.kubernetes.client.k8s_client", _fake_client),
            patch("kubernetes_asyncio.client.CoreV1Api", return_value=core),
            patch("kubernetes_asyncio.client.CustomObjectsApi", return_value=custom),
            patch("aiperf.kubernetes.console.print_cr_submission_summary"),
            patch("aiperf.kubernetes.console.save_last_benchmark"),
            patch("aiperf.kubernetes.console.print_info"),
        ):
            await deploy_via_operator(
                {"image": validated.image},
                options,
                config,
                "my-bench",
                namespace,
                dry_run=False,
                detach=True,
                no_wait=False,
                attach_port=0,
            )

        return core, custom

    async def test_explicit_preprovisioned_namespace_is_not_created(self) -> None:
        core, custom = await self._deploy(
            options=KubeOptions(
                namespace="tenant-a", image="registry.example/aiperf:latest"
            ),
            namespace="tenant-a",
        )

        assert core.created_namespaces == []
        assert custom.created_jobs == [("tenant-a", "my-bench")]

    async def test_default_cli_owned_namespace_is_created(self) -> None:
        core, custom = await self._deploy(
            options=KubeOptions(image="registry.example/aiperf:latest"),
            namespace="aiperf-benchmarks",
        )

        assert core.created_namespaces == ["aiperf-benchmarks"]
        assert custom.created_jobs == [("aiperf-benchmarks", "my-bench")]


class TestProfileModeSelection:
    """Profile preserves auto/direct behavior and offers an explicit operator path."""

    async def _run(
        self,
        *,
        operator: bool | None = None,
        no_operator: bool = False,
        discovered_operator: bool = True,
        discovery_error: BaseException | None = None,
    ) -> list[str]:
        events: list[str] = []

        async def _discover(_: KubeOptions) -> bool:
            events.append("discover")
            if discovery_error is not None:
                raise discovery_error
            return discovered_operator

        async def _operator_deploy(*_: Any, **__: Any) -> None:
            events.append("operator")

        async def _direct_deploy(*_: Any, **__: Any) -> None:
            events.append("direct")

        cli_config = MagicMock()
        cli_config.config_file = None
        config = MagicMock()
        kwargs: dict[str, Any] = {
            "cli_config": cli_config,
            "kube_options": KubeOptions(
                namespace="tenant-a", image="registry.example/aiperf:latest"
            ),
            "no_operator": no_operator,
            "detach": True,
        }
        if operator is not None:
            kwargs["operator"] = operator

        with (
            patch(
                "aiperf.cli_commands.kube.profile._resolve_spec_and_name",
                return_value=(
                    {"image": "registry.example/aiperf:latest"},
                    config,
                    "job-a",
                ),
            ),
            patch("aiperf.cli_commands.kube.profile._check_config_file_for_sweep_keys"),
            patch("aiperf.cli_commands.kube.profile._check_resolved_config_for_sweep"),
            patch("aiperf.cli_commands.kube.profile._print_memory_estimate"),
            patch(
                "aiperf.cli_commands.kube.profile_deploy.operator_available",
                _discover,
            ),
            patch(
                "aiperf.cli_commands.kube.profile_deploy.deploy_via_operator",
                _operator_deploy,
            ),
            patch(
                "aiperf.cli_commands.kube.profile_deploy_direct.deploy_direct",
                _direct_deploy,
            ),
        ):
            try:
                await profile(**kwargs)
            except TypeError as exc:
                pytest.fail(f"profile rejected the requested deployment mode: {exc}")

        return events

    async def test_explicit_operator_bypasses_crd_discovery(self) -> None:
        assert await self._run(
            operator=True,
            discovery_error=SystemExit("CRD discovery denied: HTTP 403"),
        ) == ["operator"]

    async def test_default_still_auto_detects_operator(self) -> None:
        assert await self._run(discovered_operator=True) == ["discover", "operator"]

    async def test_default_still_falls_back_to_direct_mode(self) -> None:
        assert await self._run(discovered_operator=False) == ["discover", "direct"]

    async def test_no_operator_still_forces_direct_mode(self) -> None:
        assert await self._run(no_operator=True) == ["direct"]

    async def test_conflicting_explicit_modes_are_rejected(self) -> None:
        with pytest.raises(SystemExit):
            await self._run(operator=True, no_operator=True)


class TestDeployViaOperatorPersistence:
    async def test_persists_aiperfjob_kind_for_default_targeting(self) -> None:
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _fake_client(**_: Any):
            yield MagicMock()

        validated = MagicMock()
        validated.image = "registry.example/aiperf:latest"
        validated.model_dump.return_value = {"image": validated.image}
        config = MagicMock()
        config.benchmark.endpoint.urls = []
        config.benchmark.get_model_names.return_value = ["model-a"]
        options = KubeOptions(image=validated.image, name="friendly-name")

        with (
            patch(
                "aiperf.kubernetes.spec_converter.validate_job_spec",
                return_value=validated,
            ),
            patch("aiperf.kubernetes.client.k8s_client", _fake_client),
            patch(
                "aiperf.cli_commands.kube.profile_deploy._submit_cr",
                new=AsyncMock(return_value={"spec": {"image": validated.image}}),
            ),
            patch("aiperf.kubernetes.console.print_cr_submission_summary"),
            patch("aiperf.kubernetes.console.save_last_benchmark") as mock_save,
            patch(
                "aiperf.cli_commands.kube.profile_deploy.should_detach_from_operator_job",
                return_value=True,
            ),
            patch("aiperf.kubernetes.console.print_info"),
        ):
            await deploy_via_operator(
                {"image": validated.image},
                options,
                config,
                "job-a",
                "tenant-a",
                dry_run=False,
                detach=True,
                no_wait=False,
                attach_port=0,
            )

        mock_save.assert_called_once_with(
            "job-a",
            "tenant-a",
            name="friendly-name",
            kind="AIPerfJob",
        )


# =============================================================================
# operator_available
# =============================================================================


class _StubKubeOpts:
    """Minimal KubeOptions-shaped stub for operator_available()."""

    def __init__(self) -> None:
        self.kubeconfig: str | None = None
        self.kube_context: str | None = None


class TestOperatorAvailable:
    """Tests for the CRD-existence probe."""

    async def test_returns_true_when_crd_exists(self, capsys) -> None:
        """If read_custom_resource_definition succeeds, operator mode is selected."""
        from contextlib import asynccontextmanager

        api = MagicMock()

        @asynccontextmanager
        async def _fake_client(**_kw):
            yield api

        fake_apiext = MagicMock()
        fake_apiext.read_custom_resource_definition = AsyncMock(
            return_value=MagicMock()
        )
        with (
            patch(
                "aiperf.kubernetes.client.k8s_client",
                new=_fake_client,
            ),
            patch(
                "kubernetes_asyncio.client.ApiextensionsV1Api",
                return_value=fake_apiext,
            ),
        ):
            assert await operator_available(_StubKubeOpts()) is True

        assert "operator mode" in capsys.readouterr().out

    async def test_returns_false_on_404(self, capsys) -> None:
        """404 from the API -> direct mode (no operator)."""
        from contextlib import asynccontextmanager

        from kubernetes_asyncio.client.exceptions import ApiException

        api = MagicMock()

        @asynccontextmanager
        async def _fake_client(**_kw):
            yield api

        fake_apiext = MagicMock()
        fake_apiext.read_custom_resource_definition = AsyncMock(
            side_effect=ApiException(status=404, reason="NotFound")
        )
        with (
            patch(
                "aiperf.kubernetes.client.k8s_client",
                new=_fake_client,
            ),
            patch(
                "kubernetes_asyncio.client.ApiextensionsV1Api",
                return_value=fake_apiext,
            ),
        ):
            assert await operator_available(_StubKubeOpts()) is False

        assert "no operator" in capsys.readouterr().out

    async def test_raises_on_403_rbac_denial(self) -> None:
        """403 (CRD unreadable) must NOT silently downgrade to direct mode.

        The operator may be installed but the user lacks CRD-read RBAC; we
        surface the RBAC error instead of guessing "no operator".
        """
        from contextlib import asynccontextmanager

        from kubernetes_asyncio.client.exceptions import ApiException

        api = MagicMock()

        @asynccontextmanager
        async def _fake_client(**_kw):
            yield api

        fake_apiext = MagicMock()
        fake_apiext.read_custom_resource_definition = AsyncMock(
            side_effect=ApiException(status=403, reason="Forbidden")
        )
        with (
            patch(
                "aiperf.kubernetes.client.k8s_client",
                new=_fake_client,
            ),
            patch(
                "kubernetes_asyncio.client.ApiextensionsV1Api",
                return_value=fake_apiext,
            ),
            pytest.raises(SystemExit, match="--no-operator"),
        ):
            await operator_available(_StubKubeOpts())

    async def test_raises_on_transient_5xx(self) -> None:
        """A transient apiserver 5xx must raise, not silently change mode."""
        from contextlib import asynccontextmanager

        from kubernetes_asyncio.client.exceptions import ApiException

        api = MagicMock()

        @asynccontextmanager
        async def _fake_client(**_kw):
            yield api

        fake_apiext = MagicMock()
        fake_apiext.read_custom_resource_definition = AsyncMock(
            side_effect=ApiException(status=500, reason="ServerError")
        )
        with (
            patch(
                "aiperf.kubernetes.client.k8s_client",
                new=_fake_client,
            ),
            patch(
                "kubernetes_asyncio.client.ApiextensionsV1Api",
                return_value=fake_apiext,
            ),
            pytest.raises(SystemExit, match="HTTP 500"),
        ):
            await operator_available(_StubKubeOpts())


# =============================================================================
# wait_or_detach
# =============================================================================


class TestWaitOrDetach:
    """Tests for the post-submit interactive/detach dispatcher."""

    async def test_detach_flag_prints_info_and_returns(self, capsys) -> None:
        """`detach=True` short-circuits the attach workflow."""
        opts = _StubKubeOpts()
        opts.name = "my-bench"

        with patch("sys.stdout.isatty", return_value=True):
            await wait_or_detach(
                "my-bench",
                "ns",
                opts,
                detach=True,
                no_wait=False,
                attach_port=0,
                hint="Retrieve results: aiperf kube results",
            )

        out = capsys.readouterr().out
        assert "my-bench" in out
        assert "Retrieve results" in out

    async def test_non_interactive_forces_detach_with_warning(self, capsys) -> None:
        """Non-TTY stdout auto-enables detach mode and emits a warning."""
        opts = _StubKubeOpts()
        opts.name = "bench-ci"

        attach_mock = AsyncMock()
        with (
            patch("sys.stdout.isatty", return_value=False),
            patch(
                "aiperf.kubernetes.attach.auto_attach_workflow",
                new=attach_mock,
            ),
        ):
            await wait_or_detach(
                "bench-ci",
                "ns",
                opts,
                detach=False,
                no_wait=False,
                attach_port=0,
            )

        attach_mock.assert_not_awaited()
        assert "Non-interactive" in capsys.readouterr().out

    async def test_interactive_calls_auto_attach_workflow(self) -> None:
        """Interactive + `detach=False` invokes the attach workflow."""
        opts = _StubKubeOpts()
        opts.name = "bench"

        attach_mock = AsyncMock()
        with (
            patch("sys.stdout.isatty", return_value=True),
            patch(
                "aiperf.kubernetes.attach.auto_attach_workflow",
                new=attach_mock,
            ),
        ):
            await wait_or_detach(
                "bench",
                "ns",
                opts,
                detach=False,
                no_wait=True,
                attach_port=7777,
            )

        attach_mock.assert_awaited_once()

    async def test_keyboard_interrupt_prints_interrupt_info(self, capsys) -> None:
        """Ctrl-C during attach is caught and `print_interrupt_info` fires."""
        opts = _StubKubeOpts()
        opts.name = "bench"

        with (
            patch("sys.stdout.isatty", return_value=True),
            patch(
                "aiperf.kubernetes.attach.auto_attach_workflow",
                new=AsyncMock(side_effect=KeyboardInterrupt),
            ),
        ):
            await wait_or_detach(
                "bench",
                "ns",
                opts,
                detach=False,
                no_wait=False,
                attach_port=0,
                hint="a-hint",
            )

        out = capsys.readouterr().out
        # print_interrupt_info prints job info; hint follows
        assert "a-hint" in out


# =============================================================================
# _apply_manifest
# =============================================================================


class TestApplyManifest:
    """Tests for the kind->api-call dispatch table in direct-mode deploy."""

    async def _run(self, kind: str) -> tuple[str | None, dict]:
        """Call _apply_manifest and return (label, which-client-method-was-called)."""
        manifest = {
            "kind": kind,
            "metadata": {"name": "res1", "namespace": "ns1"},
        }
        core = MagicMock()
        core.create_namespace = AsyncMock()
        core.create_namespaced_config_map = AsyncMock()
        rbac = MagicMock()
        rbac.create_namespaced_role = AsyncMock()
        rbac.create_namespaced_role_binding = AsyncMock()
        custom = MagicMock()
        custom.create_namespaced_custom_object = AsyncMock()

        label = await _apply_manifest(
            manifest, core=core, rbac=rbac, custom=custom, default_namespace="ns1"
        )
        return label, {
            "create_namespace": core.create_namespace.await_count,
            "create_configmap": core.create_namespaced_config_map.await_count,
            "create_role": rbac.create_namespaced_role.await_count,
            "create_rolebinding": rbac.create_namespaced_role_binding.await_count,
            "create_custom": custom.create_namespaced_custom_object.await_count,
        }

    @pytest.mark.parametrize(
        "kind, expected_call",
        [
            param("Namespace", "create_namespace", id="namespace"),
            param("ConfigMap", "create_configmap", id="configmap"),
            param("Role", "create_role", id="role"),
            param("RoleBinding", "create_rolebinding", id="rolebinding"),
            param("JobSet", "create_custom", id="jobset"),
        ],
    )  # fmt: skip
    async def test_kind_dispatches_to_correct_api_call(
        self, kind: str, expected_call: str
    ) -> None:
        """Each known kind routes to its matching kubernetes_asyncio call."""
        label, counts = await self._run(kind)
        assert label == f"{kind}/res1"
        assert counts[expected_call] == 1
        # Exactly one client method was called
        assert sum(counts.values()) == 1

    async def test_unknown_kind_returns_none(self) -> None:
        """Unrecognised kinds return None (no API call)."""
        label, counts = await self._run("Deployment")
        assert label is None
        assert sum(counts.values()) == 0


class TestOperatorPathDetachSemantics:
    """The operator path is the default, and it never auto-detached.

    Only the direct path called wait_or_detach, so `-d` was honored but the
    documented non-interactive auto-detach was not: `aiperf kube profile ... |
    tee log.txt` and the same command under CI blocked on watch_job for up to
    the full timeout. `--no-wait` was accepted and then deleted unused.
    """

    @pytest.mark.parametrize(
        "detach,no_wait,is_interactive,expected",
        [
            param(False, False, True, False, id="interactive-attaches"),
            param(False, False, False, True, id="piped-auto-detaches"),
            param(True, False, True, True, id="explicit-detach"),
            param(False, True, True, True, id="no-wait-returns-early"),
            param(True, True, False, True, id="all-set"),
        ],
    )  # fmt: skip
    def test_detach_matrix(
        self, detach: bool, no_wait: bool, is_interactive: bool, expected: bool
    ) -> None:
        from aiperf.cli_commands.kube.profile_deploy import (
            should_detach_from_operator_job,
        )

        assert (
            should_detach_from_operator_job(
                detach=detach, no_wait=no_wait, is_interactive=is_interactive
            )
            is expected
        )
