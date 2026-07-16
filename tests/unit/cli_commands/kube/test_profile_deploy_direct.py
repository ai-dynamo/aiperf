# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for `aiperf.cli_commands.kube.profile_deploy_direct`.

Companion to ``test_profile_deploy.py`` (which already covers ``_apply_manifest``
and ``wait_or_detach``). This file exercises the helpers that wrap the direct-
deploy (no operator) flow:

- ``_prepare_direct_deploy`` — applies K8s overlays, computes pod count
- ``_apply_all_manifests`` — opens the API, dispatches per manifest, treats
  409 (AlreadyExists) as success and surfaces other ApiExceptions
- ``deploy_direct`` — full glue: dry-run prints YAML, wet-run creates resources
  and saves the last-benchmark hint, then hands off to ``wait_or_detach``
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_kube_options(image: str = "aiperf:latest", **overrides: Any) -> Any:
    """Build a minimal KubeOptions object for direct-deploy."""
    from aiperf.config.kube import KubeOptions

    return KubeOptions(image=image, **overrides)


def _make_config(concurrency: int | None = 8) -> MagicMock:
    """Build a stub AIPerfConfig-like object the helpers read from."""
    phase = MagicMock()
    phase.concurrency = concurrency
    config = MagicMock()
    config.benchmark.phases = [phase]
    config.benchmark.endpoint.urls = ["http://svc:8000/v1/chat/completions"]
    config.benchmark.get_model_names.return_value = ["test-model"]
    # model_dump (used by _prepare_direct_deploy before apply_k8s_runtime_config)
    config.model_dump.return_value = {"phases": [{"concurrency": concurrency}]}
    return config


@asynccontextmanager
async def _fake_k8s_client(**_kw: Any):
    yield MagicMock(name="ApiClient")


# ---------------------------------------------------------------------------
# _prepare_direct_deploy
# ---------------------------------------------------------------------------


class TestPrepareDirectDeploy:
    """Overlay K8s-runtime config + worker count math."""

    def test_basic_flow_returns_three_tuple(self) -> None:
        from aiperf.cli_commands.kube.profile_deploy_direct import (
            _prepare_direct_deploy,
        )

        revalidated = MagicMock()
        revalidated_phase = MagicMock()
        revalidated_phase.concurrency = 8
        revalidated.benchmark.phases = [revalidated_phase]

        deploy_config = MagicMock()
        deploy_config.connections_per_worker = 4

        kube_opts = _make_kube_options()
        # Simulate not having ttl_seconds in user-set fields so we trigger the
        # default-bump branch.
        with (
            patch(
                "aiperf.config.AIPerfConfig.model_validate",
                return_value=revalidated,
            ),
            patch(
                "aiperf.operator.spec_converter.apply_k8s_runtime_config"
            ) as mock_overlay,
            patch(
                "aiperf.operator.spec_converter.apply_worker_config", return_value=2
            ) as mock_apply_workers,
            patch(
                "aiperf.config.kube.KubeOptions.to_deployment_config",
                return_value=deploy_config,
            ),
        ):
            cfg, dc, num_pods = _prepare_direct_deploy(
                _make_config(concurrency=8), kube_opts, "bench", "ns"
            )

        assert cfg is revalidated
        assert dc is deploy_config
        assert num_pods == 2
        # Overlay was applied to the dumped dict
        mock_overlay.assert_called_once()
        # apply_worker_config received total_workers = ceil(8/4) = 2
        mock_apply_workers.assert_called_once_with(revalidated, 2)

    def test_default_ttl_bumped_when_user_unset(self) -> None:
        """When --ttl-seconds is unset, direct-mode bumps to DIRECT_MODE_TTL_SECONDS."""
        from aiperf.cli_commands.kube.profile_deploy_direct import (
            _prepare_direct_deploy,
        )
        from aiperf.kubernetes.environment import K8sEnvironment

        revalidated_phase = MagicMock()
        revalidated_phase.concurrency = 1
        revalidated = MagicMock()
        revalidated.phases = [revalidated_phase]

        deploy_config = MagicMock()
        deploy_config.connections_per_worker = 1
        deploy_config.ttl_seconds_after_finished = 300

        kube_opts = _make_kube_options()
        # User did NOT set ttl_seconds explicitly
        assert "ttl_seconds" not in kube_opts.model_fields_set

        with (
            patch(
                "aiperf.config.AIPerfConfig.model_validate", return_value=revalidated
            ),
            patch("aiperf.operator.spec_converter.apply_k8s_runtime_config"),
            patch("aiperf.operator.spec_converter.apply_worker_config", return_value=1),
            patch(
                "aiperf.config.kube.KubeOptions.to_deployment_config",
                return_value=deploy_config,
            ),
        ):
            _prepare_direct_deploy(_make_config(), kube_opts, "bench", "ns")

        assert (
            deploy_config.ttl_seconds_after_finished
            == K8sEnvironment.JOBSET.DIRECT_MODE_TTL_SECONDS
        )

    def test_explicit_ttl_seconds_not_bumped(self) -> None:
        """When --ttl-seconds=N is explicit, direct mode preserves it."""
        from aiperf.cli_commands.kube.profile_deploy_direct import (
            _prepare_direct_deploy,
        )

        revalidated_phase = MagicMock()
        revalidated_phase.concurrency = 1
        revalidated = MagicMock()
        revalidated.phases = [revalidated_phase]

        deploy_config = MagicMock()
        deploy_config.connections_per_worker = 1
        deploy_config.ttl_seconds_after_finished = 120

        kube_opts = _make_kube_options(ttl_seconds=120)
        # User DID set ttl_seconds explicitly
        assert "ttl_seconds" in kube_opts.model_fields_set

        with (
            patch(
                "aiperf.config.AIPerfConfig.model_validate", return_value=revalidated
            ),
            patch("aiperf.operator.spec_converter.apply_k8s_runtime_config"),
            patch("aiperf.operator.spec_converter.apply_worker_config", return_value=1),
            patch(
                "aiperf.config.kube.KubeOptions.to_deployment_config",
                return_value=deploy_config,
            ),
        ):
            _prepare_direct_deploy(_make_config(), kube_opts, "bench", "ns")

        # The user-set 120 must not be overwritten
        assert deploy_config.ttl_seconds_after_finished == 120

    @pytest.mark.parametrize(
        "concurrency,connections_per_worker,expected_workers",
        [
            (1, 1, 1),
            (10, 10, 1),
            (10, 4, 3),
            (1, 100, 1),
            (0, 5, 1),  # concurrency=0 falls back to 1
        ],
    )  # fmt: skip
    def test_worker_count_math(
        self,
        concurrency: int,
        connections_per_worker: int,
        expected_workers: int,
    ) -> None:
        """total_workers = max(1, ceil(concurrency / connections_per_worker))."""
        from aiperf.cli_commands.kube.profile_deploy_direct import (
            _prepare_direct_deploy,
        )

        revalidated_phase = MagicMock()
        revalidated_phase.concurrency = concurrency
        revalidated = MagicMock()
        revalidated.benchmark.phases = [revalidated_phase]

        deploy_config = MagicMock()
        deploy_config.connections_per_worker = connections_per_worker
        deploy_config.ttl_seconds_after_finished = 0

        kube_opts = _make_kube_options(ttl_seconds=10)

        with (
            patch(
                "aiperf.config.AIPerfConfig.model_validate", return_value=revalidated
            ),
            patch("aiperf.operator.spec_converter.apply_k8s_runtime_config"),
            patch(
                "aiperf.operator.spec_converter.apply_worker_config",
                side_effect=lambda _cfg, n: n,
            ),
            patch(
                "aiperf.config.kube.KubeOptions.to_deployment_config",
                return_value=deploy_config,
            ),
        ):
            _, _, num_pods = _prepare_direct_deploy(
                _make_config(concurrency=concurrency), kube_opts, "bench", "ns"
            )

        assert num_pods == expected_workers


# ---------------------------------------------------------------------------
# _apply_all_manifests
# ---------------------------------------------------------------------------


class TestApplyAllManifests:
    """Open k8s_client, dispatch each manifest, handle 409, surface other errors."""

    @pytest.mark.asyncio
    async def test_creates_each_manifest_and_logs_success(self) -> None:
        from aiperf.cli_commands.kube.profile_deploy_direct import _apply_all_manifests

        manifests = [
            {"kind": "Namespace", "metadata": {"name": "ns1"}},
            {"kind": "ConfigMap", "metadata": {"name": "cm1"}},
        ]
        opts = _make_kube_options()

        with (
            patch("aiperf.kubernetes.client.k8s_client", new=_fake_k8s_client),
            patch(
                "aiperf.cli_commands.kube.profile_deploy_direct._apply_manifest",
                new=AsyncMock(side_effect=["Namespace/ns1", "ConfigMap/cm1"]),
            ) as mock_apply,
            patch("aiperf.kubernetes.console.print_success") as mock_success,
        ):
            await _apply_all_manifests(manifests, opts, effective_ns="ns1")

        assert mock_apply.await_count == 2
        assert mock_success.call_count == 2

    @pytest.mark.asyncio
    async def test_409_already_exists_is_logged_not_raised(self) -> None:
        """A 409 ApiException -> print_info('already exists') and continue."""
        from kubernetes_asyncio.client.exceptions import ApiException

        from aiperf.cli_commands.kube.profile_deploy_direct import _apply_all_manifests

        manifests = [
            {"kind": "ConfigMap", "metadata": {"name": "cm1"}},
            {"kind": "ConfigMap", "metadata": {"name": "cm2"}},
        ]
        opts = _make_kube_options()

        side_effects: list[Any] = [
            ApiException(status=409, reason="AlreadyExists"),
            "ConfigMap/cm2",
        ]
        with (
            patch("aiperf.kubernetes.client.k8s_client", new=_fake_k8s_client),
            patch(
                "aiperf.cli_commands.kube.profile_deploy_direct._apply_manifest",
                new=AsyncMock(side_effect=side_effects),
            ),
            patch("aiperf.kubernetes.console.print_info") as mock_info,
            patch("aiperf.kubernetes.console.print_success") as mock_success,
        ):
            await _apply_all_manifests(manifests, opts, effective_ns="ns1")

        # First manifest: 409 -> print_info, no success
        # Second manifest: success
        assert mock_info.call_count == 1
        assert "already exists" in mock_info.call_args.args[0]
        assert mock_success.call_count == 1

    @pytest.mark.asyncio
    async def test_non_409_api_exception_surfaces(self) -> None:
        """Status != 409 must raise out of _apply_all_manifests."""
        from kubernetes_asyncio.client.exceptions import ApiException

        from aiperf.cli_commands.kube.profile_deploy_direct import _apply_all_manifests

        manifests = [{"kind": "ConfigMap", "metadata": {"name": "cm1"}}]
        opts = _make_kube_options()

        with (
            patch("aiperf.kubernetes.client.k8s_client", new=_fake_k8s_client),
            patch(
                "aiperf.cli_commands.kube.profile_deploy_direct._apply_manifest",
                new=AsyncMock(side_effect=ApiException(status=500, reason="Boom")),
            ),
            pytest.raises(ApiException) as exc_info,
        ):
            await _apply_all_manifests(manifests, opts, effective_ns="ns1")

        assert exc_info.value.status == 500

    @pytest.mark.asyncio
    async def test_unknown_kind_logs_warning_and_continues(self) -> None:
        """label=None from _apply_manifest -> print_warning, no print_success."""
        from aiperf.cli_commands.kube.profile_deploy_direct import _apply_all_manifests

        manifests = [{"kind": "Pod", "metadata": {"name": "p1"}}]
        opts = _make_kube_options()

        with (
            patch("aiperf.kubernetes.client.k8s_client", new=_fake_k8s_client),
            patch(
                "aiperf.cli_commands.kube.profile_deploy_direct._apply_manifest",
                new=AsyncMock(return_value=None),
            ),
            patch("aiperf.kubernetes.console.print_warning") as mock_warn,
            patch("aiperf.kubernetes.console.print_success") as mock_success,
        ):
            await _apply_all_manifests(manifests, opts, effective_ns="ns1")

        mock_warn.assert_called_once()
        assert "Unknown resource kind" in mock_warn.call_args.args[0]
        mock_success.assert_not_called()

    @pytest.mark.asyncio
    async def test_kubeconfig_and_context_propagate_to_k8s_client(self) -> None:
        """k8s_client is opened with kubeconfig + context from KubeOptions."""
        from aiperf.cli_commands.kube.profile_deploy_direct import _apply_all_manifests

        captured: dict[str, Any] = {}

        @asynccontextmanager
        async def _capturing_client(**kwargs: Any):
            captured.update(kwargs)
            yield MagicMock()

        opts = _make_kube_options(kubeconfig="/tmp/kc", kube_context="ctx-1")
        with (
            patch("aiperf.kubernetes.client.k8s_client", new=_capturing_client),
            patch(
                "aiperf.cli_commands.kube.profile_deploy_direct._apply_manifest",
                new=AsyncMock(return_value=None),
            ),
            patch("aiperf.kubernetes.console.print_warning"),
        ):
            await _apply_all_manifests(
                [{"kind": "Pod", "metadata": {"name": "p1"}}],
                opts,
                effective_ns="ns1",
            )

        assert captured.get("kubeconfig") == "/tmp/kc"
        assert captured.get("context") == "ctx-1"


# ---------------------------------------------------------------------------
# deploy_direct
# ---------------------------------------------------------------------------


class TestDeployDirect:
    """End-to-end glue for the direct-mode deploy path."""

    @pytest.mark.asyncio
    async def test_dry_run_prints_yaml_and_skips_apply(self) -> None:
        """dry_run=True -> _print_manifests_yaml is called, no API ops, no wait."""
        from aiperf.cli_commands.kube.profile_deploy_direct import deploy_direct

        deploy_config = MagicMock(connections_per_worker=4)
        deployment = MagicMock()
        deployment.effective_namespace = "eff-ns"
        deployment.get_all_manifests.return_value = [{"kind": "Namespace"}]

        with (
            patch(
                "aiperf.cli_commands.kube.profile_deploy_direct._prepare_direct_deploy",
                return_value=(_make_config(), deploy_config, 1),
            ),
            patch(
                "aiperf.kubernetes.resources.KubernetesDeployment",
                return_value=deployment,
            ),
            patch(
                "aiperf.cli_commands.kube.profile_deploy_direct._print_manifests_yaml"
            ) as mock_print_yaml,
            patch(
                "aiperf.cli_commands.kube.profile_deploy_direct._apply_all_manifests",
                new=AsyncMock(),
            ) as mock_apply_all,
            patch(
                "aiperf.cli_commands.kube.profile_deploy_direct.wait_or_detach",
                new=AsyncMock(),
            ) as mock_wait,
        ):
            await deploy_direct(
                _make_config(),
                _make_kube_options(),
                "bench",
                "ns",
                dry_run=True,
                detach=False,
                no_wait=False,
                attach_port=0,
            )

        mock_print_yaml.assert_called_once()
        mock_apply_all.assert_not_awaited()
        mock_wait.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_full_deploy_flow_calls_apply_all_and_waits(self) -> None:
        """Wet-run path: apply manifests, save last-benchmark, then wait_or_detach."""
        from aiperf.cli_commands.kube.profile_deploy_direct import deploy_direct

        deploy_config = MagicMock(connections_per_worker=4)
        deployment = MagicMock()
        deployment.effective_namespace = "eff-ns"
        manifests = [
            {"kind": "Namespace", "metadata": {"name": "eff-ns"}},
            {"kind": "JobSet", "metadata": {"name": "bench"}},
        ]
        deployment.get_all_manifests.return_value = manifests

        opts = _make_kube_options(image="custom:42")

        with (
            patch(
                "aiperf.cli_commands.kube.profile_deploy_direct._prepare_direct_deploy",
                return_value=(_make_config(), deploy_config, 3),
            ),
            patch(
                "aiperf.kubernetes.resources.KubernetesDeployment",
                return_value=deployment,
            ),
            patch(
                "aiperf.cli_commands.kube.profile_deploy_direct._apply_all_manifests",
                new=AsyncMock(),
            ) as mock_apply_all,
            patch(
                "aiperf.kubernetes.console.print_cr_submission_summary"
            ) as mock_summary,
            patch("aiperf.kubernetes.console.save_last_benchmark") as mock_save,
            patch(
                "aiperf.cli_commands.kube.profile_deploy_direct.wait_or_detach",
                new=AsyncMock(),
            ) as mock_wait,
        ):
            await deploy_direct(
                _make_config(),
                opts,
                "bench",
                "ns",
                dry_run=False,
                detach=False,
                no_wait=False,
                attach_port=0,
            )

        mock_apply_all.assert_awaited_once()
        # First positional: manifests; second: kube_options; effective_ns kwarg
        apply_args = mock_apply_all.await_args
        assert apply_args.args[0] == manifests
        assert apply_args.args[1] is opts
        assert apply_args.kwargs.get("effective_ns") == "eff-ns"

        # CR submission summary uses the EFFECTIVE namespace (not the input)
        mock_summary.assert_called_once()
        assert mock_summary.call_args.kwargs.get("namespace") == "eff-ns"
        assert mock_summary.call_args.kwargs.get("image") == "custom:42"

        # Last-benchmark hint is saved with effective namespace
        mock_save.assert_called_once_with("bench", "eff-ns", name=opts.name)

        # And the deploy hands off to wait_or_detach with the right hint
        mock_wait.assert_awaited_once()
        assert "aiperf kube results" in mock_wait.await_args.kwargs.get("hint", "")

    @pytest.mark.asyncio
    async def test_partial_failure_in_apply_all_propagates(self) -> None:
        """A failure inside _apply_all_manifests must surface; wait_or_detach skipped."""
        from aiperf.cli_commands.kube.profile_deploy_direct import deploy_direct

        deploy_config = MagicMock(connections_per_worker=1)
        deployment = MagicMock()
        deployment.effective_namespace = "ns"
        deployment.get_all_manifests.return_value = [{"kind": "Namespace"}]

        with (
            patch(
                "aiperf.cli_commands.kube.profile_deploy_direct._prepare_direct_deploy",
                return_value=(_make_config(), deploy_config, 1),
            ),
            patch(
                "aiperf.kubernetes.resources.KubernetesDeployment",
                return_value=deployment,
            ),
            patch(
                "aiperf.cli_commands.kube.profile_deploy_direct._apply_all_manifests",
                new=AsyncMock(side_effect=RuntimeError("permission denied")),
            ),
            patch("aiperf.kubernetes.console.save_last_benchmark") as mock_save,
            patch(
                "aiperf.cli_commands.kube.profile_deploy_direct.wait_or_detach",
                new=AsyncMock(),
            ) as mock_wait,
            pytest.raises(RuntimeError, match="permission denied"),
        ):
            await deploy_direct(
                _make_config(),
                _make_kube_options(),
                "bench",
                "ns",
                dry_run=False,
                detach=False,
                no_wait=False,
                attach_port=0,
            )

        # Last-benchmark hint and wait_or_detach must NOT run after a deploy failure
        mock_save.assert_not_called()
        mock_wait.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_namespace_propagates_via_kube_options(self) -> None:
        """KubernetesDeployment receives kube_options.namespace verbatim."""
        from aiperf.cli_commands.kube.profile_deploy_direct import deploy_direct

        deploy_config = MagicMock(connections_per_worker=1)
        deployment = MagicMock()
        deployment.effective_namespace = "passed-ns"
        deployment.get_all_manifests.return_value = []

        opts = _make_kube_options(namespace="passed-ns")
        captured: dict[str, Any] = {}

        def _factory(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return deployment

        with (
            patch(
                "aiperf.cli_commands.kube.profile_deploy_direct._prepare_direct_deploy",
                return_value=(_make_config(), deploy_config, 1),
            ),
            patch(
                "aiperf.kubernetes.resources.KubernetesDeployment",
                _factory,
            ),
            patch(
                "aiperf.cli_commands.kube.profile_deploy_direct._apply_all_manifests",
                new=AsyncMock(),
            ),
            patch("aiperf.kubernetes.console.print_cr_submission_summary"),
            patch("aiperf.kubernetes.console.save_last_benchmark"),
            patch(
                "aiperf.cli_commands.kube.profile_deploy_direct.wait_or_detach",
                new=AsyncMock(),
            ),
        ):
            await deploy_direct(
                _make_config(),
                opts,
                "bench",
                "ignored-input-ns",
                dry_run=False,
                detach=False,
                no_wait=False,
                attach_port=0,
            )

        # The deployment receives kube_options.namespace, NOT the positional 'namespace' arg
        assert captured.get("namespace") == "passed-ns"
        assert captured.get("job_id") == "bench"
        assert captured.get("worker_replicas") == 1
        # Model names come from config.benchmark.get_model_names()
        assert captured.get("model_names") == ["test-model"]

    @pytest.mark.asyncio
    async def test_image_override_appears_in_submission_summary(self) -> None:
        """KubeOptions.image is forwarded to print_cr_submission_summary."""
        from aiperf.cli_commands.kube.profile_deploy_direct import deploy_direct

        deploy_config = MagicMock(connections_per_worker=1)
        deployment = MagicMock()
        deployment.effective_namespace = "ns"
        deployment.get_all_manifests.return_value = []

        opts = _make_kube_options(image="ghcr.io/aiperf:override")

        with (
            patch(
                "aiperf.cli_commands.kube.profile_deploy_direct._prepare_direct_deploy",
                return_value=(_make_config(), deploy_config, 1),
            ),
            patch(
                "aiperf.kubernetes.resources.KubernetesDeployment",
                return_value=deployment,
            ),
            patch(
                "aiperf.cli_commands.kube.profile_deploy_direct._apply_all_manifests",
                new=AsyncMock(),
            ),
            patch(
                "aiperf.kubernetes.console.print_cr_submission_summary"
            ) as mock_summary,
            patch("aiperf.kubernetes.console.save_last_benchmark"),
            patch(
                "aiperf.cli_commands.kube.profile_deploy_direct.wait_or_detach",
                new=AsyncMock(),
            ),
        ):
            await deploy_direct(
                _make_config(),
                opts,
                "bench",
                "ns",
                dry_run=False,
                detach=False,
                no_wait=False,
                attach_port=0,
            )

        assert mock_summary.call_args.kwargs.get("image") == "ghcr.io/aiperf:override"

    @pytest.mark.asyncio
    async def test_skip_endpoint_check_param_accepted_and_ignored(self) -> None:
        """``skip_endpoint_check`` is accepted for CLI parity but never read."""
        from aiperf.cli_commands.kube.profile_deploy_direct import deploy_direct

        deploy_config = MagicMock(connections_per_worker=1)
        deployment = MagicMock()
        deployment.effective_namespace = "ns"
        deployment.get_all_manifests.return_value = []

        with (
            patch(
                "aiperf.cli_commands.kube.profile_deploy_direct._prepare_direct_deploy",
                return_value=(_make_config(), deploy_config, 1),
            ),
            patch(
                "aiperf.kubernetes.resources.KubernetesDeployment",
                return_value=deployment,
            ),
            patch(
                "aiperf.cli_commands.kube.profile_deploy_direct._apply_all_manifests",
                new=AsyncMock(),
            ),
            patch("aiperf.kubernetes.console.print_cr_submission_summary"),
            patch("aiperf.kubernetes.console.save_last_benchmark"),
            patch(
                "aiperf.cli_commands.kube.profile_deploy_direct.wait_or_detach",
                new=AsyncMock(),
            ),
        ):
            # Both True and False flow through without raising
            await deploy_direct(
                _make_config(),
                _make_kube_options(),
                "bench",
                "ns",
                dry_run=False,
                detach=True,
                no_wait=False,
                attach_port=0,
                skip_endpoint_check=True,
            )

    @pytest.mark.asyncio
    async def test_real_config_model_names_reach_deployment(self) -> None:
        """Attr-path drift guard using a REAL AIPerfConfig, not a MagicMock.

        MagicMock configs auto-create whatever attribute path the source reads
        (``config.get_model_names`` vs ``config.benchmark.get_model_names``
        both "work"), so path drift silently no-ops. Here a real config flows
        through the unmocked ``_prepare_direct_deploy`` and the model names
        read by ``deploy_direct`` must land in the KubernetesDeployment params.
        """
        from aiperf.cli_commands.kube.profile_deploy_direct import deploy_direct
        from aiperf.config import AIPerfConfig

        config = AIPerfConfig.model_validate(
            {
                "benchmark": {
                    "models": ["test-model"],
                    "endpoint": {
                        "urls": ["http://svc:8000/v1/chat/completions"],
                        "type": "chat",
                    },
                    "datasets": [
                        {
                            "name": "default",
                            "type": "synthetic",
                            "entries": 10,
                            "prompts": {"isl": 64, "osl": 16},
                        }
                    ],
                    "phases": [
                        {
                            "name": "default",
                            "type": "concurrency",
                            "concurrency": 2,
                            "requests": 10,
                        }
                    ],
                }
            }
        )
        assert config.benchmark.get_model_names() == ["test-model"]

        deployment = MagicMock()
        deployment.effective_namespace = "ns"
        deployment.get_all_manifests.return_value = []
        captured: dict[str, Any] = {}

        def _factory(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return deployment

        # _prepare_direct_deploy runs UNMOCKED: model_dump -> k8s overlay ->
        # AIPerfConfig.model_validate -> apply_worker_config, all on the real
        # config. dry_run=True keeps the K8s API out of the picture.
        with (
            patch("aiperf.kubernetes.resources.KubernetesDeployment", _factory),
            patch(
                "aiperf.cli_commands.kube.profile_deploy_direct._print_manifests_yaml"
            ),
        ):
            await deploy_direct(
                config,
                _make_kube_options(),
                "bench",
                "ns",
                dry_run=True,
                detach=False,
                no_wait=False,
                attach_port=0,
            )

        assert captured.get("model_names") == ["test-model"]
