# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for Kubernetes sweep-facing CLI contracts.

Focuses on:
- AIPerfSweep list/detail resolver behavior when jobs and sweeps overlap or vanish.
- `-v` / `-t` child-selector validation before logs/results touch Kubernetes.
- namespace, kubeconfig, and kube-context propagation for sweep submission/downloads.
- JSON/text output cleanliness for machine-readable sweep-child run listing.
- URL and local-path encoding for sweep aggregate artifacts.

Out of scope: live apiserver behavior, real port-forwards, and transfer-body retries;
those are covered by the Kubernetes client and results-operator suites.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, get_args, get_type_hints
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest
from cyclopts import Parameter
from pytest import param

from aiperf.cli_commands.kube._kube_common import resolve_child_name
from aiperf.cli_commands.kube.attach import attach as attach_command
from aiperf.cli_commands.kube.cancel import cancel as cancel_command
from aiperf.cli_commands.kube.debug import debug as debug_command
from aiperf.cli_commands.kube.logs import logs as logs_command
from aiperf.cli_commands.kube.results import list_runs as list_runs_command
from aiperf.cli_commands.kube.results import results as results_command
from aiperf.config.flags import CLIConfig
from aiperf.config.kube import KubeManageOptions, KubeOptions
from aiperf.kubernetes.models import AIPerfSweepInfo

# ============================================================
# Helpers
# ============================================================


class _FakeApiClient:
    """Open API-client sentinel with observable close semantics."""

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _Response:
    """Tiny aiohttp response stand-in for URL-shape tests."""

    def __init__(self, *, body: bytes = b"{}", status: int = 200) -> None:
        self.status = status
        self.headers: dict[str, str] = {}
        self.content = self
        self._body = body

    async def __aenter__(self) -> _Response:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def iter_chunked(self, _size: int) -> AsyncIterator[bytes]:
        yield self._body

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")


class _Session:
    """Session stub that records requested URLs and returns one response."""

    def __init__(self, response: _Response) -> None:
        self.response = response
        self.requested: list[tuple[str, dict[str, str]]] = []

    def get(self, url: str, **kwargs: object) -> _Response:
        headers = kwargs.get("headers")
        self.requested.append((url, headers if isinstance(headers, dict) else {}))
        return self.response


class _ResolvedSweepStub:
    """ResolvedSweep-like object consumed by `_run_sweep_results`."""

    def __init__(self) -> None:
        self.name = "latency-sweep"
        self.namespace = "bench-prod"
        self.api = _FakeApiClient()

    async def aclose(self) -> None:
        await self.api.close()


@asynccontextmanager
async def _fake_k8s_client(**kwargs: object) -> AsyncIterator[MagicMock]:
    api = MagicMock(name="ApiClient")
    api.kwargs = kwargs
    yield api


def _sweep_info(
    *, name: str = "latency-sweep", namespace: str = "bench-prod"
) -> AIPerfSweepInfo:
    return AIPerfSweepInfo(
        name=name,
        namespace=namespace,
        phase="Running",
        run_epoch=1770001234,
        total_variations=4,
        max_total_runs=8,
        completed_runs=2,
        failed_runs=0,
        created="2026-05-18T12:00:00Z",
    )


def _api_exception(status: int):
    from kubernetes_asyncio.client.exceptions import ApiException

    return ApiException(status=status, reason=f"status-{status}")


# ============================================================
# Child-selector validation
# ============================================================


class TestSweepSelectorValidation:
    """Variation/trial selectors must fail before silently retargeting a parent."""

    @pytest.mark.parametrize(
        ("parent", "variation", "trial", "expected"),
        [
            ("latency-sweep", 0, None, "latency-sweep-v00"),
            ("latency-sweep", 7, 2, "latency-sweep-v07-t2"),
            param("latency-sweep", 199, 9, "latency-sweep-v199-t9", id="max-boundaries"),
        ],
    )  # fmt: skip
    def test_resolve_child_name_valid_selector_returns_child_job_name(
        self, parent: str, variation: int, trial: int | None, expected: str
    ) -> None:
        assert resolve_child_name(parent, variation=variation, trial=trial) == expected

    @pytest.mark.parametrize(
        "command",
        [
            param(attach_command, id="attach"),
            param(cancel_command, id="cancel"),
            param(debug_command, id="debug"),
            param(logs_command, id="logs"),
            param(results_command, id="results"),
            param(list_runs_command, id="results-list-runs"),
        ],
    )
    def test_variation_help_matches_supported_selector_range(
        self, command: Any
    ) -> None:
        annotation = get_type_hints(command, include_extras=True)["variation"]
        parameter = next(
            item for item in get_args(annotation) if isinstance(item, Parameter)
        )

        assert parameter.help is not None
        assert "variation index (0..199)" in parameter.help

    @pytest.mark.parametrize(
        ("variation", "trial", "match"),
        [
            param(None, 0, r"trial.*requires.*variation", id="trial-without-variation"),
            param(-1, None, r"variation.*0.*199", id="negative-variation"),
            param(200, None, r"variation.*0.*199", id="variation-above-max"),
            param(7, -1, r"trial.*0.*9", id="negative-trial"),
            param(7, 10, r"trial.*0.*9", id="trial-above-max"),
        ],
    )  # fmt: skip
    def test_resolve_child_name_invalid_selector_raises_before_lookup(
        self, variation: int | None, trial: int | None, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            resolve_child_name("latency-sweep", variation=variation, trial=trial)

    @pytest.mark.asyncio
    async def test_results_trial_without_variation_rejects_before_parent_download(
        self, tmp_path: Path
    ) -> None:
        from aiperf.cli_commands.kube.results import results

        with (
            patch(
                "aiperf.cli_commands.kube.results._run_results",
                new=AsyncMock(),
            ) as mock_run,
            pytest.raises(ValueError, match=r"trial.*requires.*variation"),
        ):
            await results(
                job_id="latency-sweep",
                manage_options=KubeManageOptions(namespace="bench-prod"),
                output=tmp_path / "child-results",
                trial=0,
            )

        mock_run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_logs_trial_without_variation_rejects_before_parent_lookup(
        self,
    ) -> None:
        from aiperf.cli_commands.kube.logs import logs

        with (
            patch(
                "aiperf.kubernetes.cli_helpers.resolve_job_id_and_namespace",
                return_value=("latency-sweep", "bench-prod"),
            ) as mock_resolve,
            pytest.raises(ValueError, match=r"trial.*requires.*variation"),
        ):
            await logs(
                job_id="latency-sweep",
                manage_options=KubeManageOptions(namespace="bench-prod"),
                trial=0,
            )

        mock_resolve.assert_not_called()


# ============================================================
# Sweep list/detail resolution and missing-sweep behavior
# ============================================================


class TestSweepListDetailResolution:
    """Sweep-aware lookup names both resource kinds and avoids cross-namespace leaks."""

    @pytest.mark.asyncio
    async def test_resolve_target_job_miss_then_sweep_hit_returns_resolved_sweep(
        self,
    ) -> None:
        from aiperf.kubernetes import cli_helpers

        api = _FakeApiClient()
        sweep = _sweep_info(namespace="tenant-a")
        with (
            patch(
                "aiperf.kubernetes.cli_helpers._open_api_client",
                new=AsyncMock(return_value=api),
            ) as mock_open,
            patch(
                "aiperf.kubernetes.client.find_aiperf_job",
                new=AsyncMock(return_value=None),
            ) as mock_job,
            patch(
                "aiperf.kubernetes.client.find_aiperf_sweep",
                new=AsyncMock(return_value=sweep),
            ) as mock_sweep,
            patch(
                "aiperf.kubernetes.client.find_jobset",
                new=AsyncMock(),
            ) as mock_jobset,
        ):
            resolved = await cli_helpers.resolve_target(
                "latency-sweep",
                namespace="tenant-a",
                kubeconfig="/secure/kubeconfigs/dgx-prod.yaml",
                kube_context="dgx-prod-admin",
            )

        assert resolved is not None
        assert resolved.name == "latency-sweep"
        assert resolved.namespace == "tenant-a"
        assert api.closed is False
        assert mock_open.await_args.kwargs == {
            "kubeconfig": "/secure/kubeconfigs/dgx-prod.yaml",
            "kube_context": "dgx-prod-admin",
        }
        assert mock_job.await_args.args[1:] == ("latency-sweep", "tenant-a")
        assert mock_sweep.await_args.args[1:] == ("latency-sweep", "tenant-a")
        mock_jobset.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_resolve_target_missing_sweep_prints_both_kinds_and_closes_api(
        self,
    ) -> None:
        from aiperf.kubernetes import cli_helpers

        api = _FakeApiClient()
        with (
            patch(
                "aiperf.kubernetes.cli_helpers._open_api_client",
                new=AsyncMock(return_value=api),
            ),
            patch(
                "aiperf.kubernetes.client.find_aiperf_job",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "aiperf.kubernetes.client.find_aiperf_sweep",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "aiperf.kubernetes.client.find_jobset",
                new=AsyncMock(return_value=None),
            ),
            patch("aiperf.kubernetes.cli_helpers.print_error") as mock_error,
            patch("aiperf.kubernetes.cli_helpers.print_info") as mock_info,
        ):
            resolved = await cli_helpers.resolve_target(
                "missing-sweep", namespace="tenant-a"
            )

        assert resolved is None
        assert api.closed is True
        assert "AIPerfJob or AIPerfSweep" in mock_error.call_args.args[0]
        assert "missing-sweep" in mock_error.call_args.args[0]
        assert any("tenant-a" in call.args[0] for call in mock_info.call_args_list)

    @pytest.mark.asyncio
    async def test_list_aiperfsweeps_namespace_required_when_not_all_namespaces(
        self,
    ) -> None:
        from aiperf.kubernetes.client import list_aiperfsweeps

        with pytest.raises(ValueError, match=r"namespace.*all_namespaces"):
            await list_aiperfsweeps(MagicMock(), namespace=None, all_namespaces=False)

    @pytest.mark.asyncio
    async def test_find_aiperfsweep_404_returns_none_without_cluster_fallback(
        self,
    ) -> None:
        from aiperf.kubernetes.client import find_aiperfsweep

        custom = MagicMock()
        custom.get_namespaced_custom_object = AsyncMock(side_effect=_api_exception(404))
        with patch("kubernetes_asyncio.client.CustomObjectsApi", return_value=custom):
            found = await find_aiperfsweep(MagicMock(), "tenant-a", "missing-sweep")

        assert found is None
        custom.get_namespaced_custom_object.assert_awaited_once()


# ============================================================
# Namespace, kube-context propagation and output cleanliness
# ============================================================


class TestSweepCommandPropagationAndOutput:
    """Sweep CLI commands carry kube credentials to every boundary they open."""

    @pytest.mark.asyncio
    async def test_submit_sweep_uses_namespace_context_and_sweep_plural(self) -> None:
        from aiperf.cli_commands.kube.sweep import _submit_sweep

        custom = MagicMock()
        custom.create_namespaced_custom_object = AsyncMock(return_value={})
        cr = {
            "apiVersion": "aiperf.nvidia.com/v1alpha1",
            "kind": "AIPerfSweep",
            "metadata": {"name": "latency-sweep"},
            "spec": {"image": "aiperf:branch"},
        }
        opts = KubeOptions(
            image="aiperf:branch",
            kubeconfig="/secure/kubeconfigs/dgx-prod.yaml",
            kube_context="dgx-prod-admin",
        )
        with (
            patch(
                "aiperf.kubernetes.client.k8s_client", new=_fake_k8s_client
            ) as _client,
            patch("kubernetes_asyncio.client.CustomObjectsApi", return_value=custom),
            patch("aiperf.kubernetes.console.console"),
            patch("aiperf.kubernetes.console.save_last_benchmark") as mock_save,
        ):
            await _submit_sweep(cr_dict=cr, kube_options=opts, namespace="tenant-a")

        assert cr["metadata"]["namespace"] == "tenant-a"
        assert custom.create_namespaced_custom_object.await_args.kwargs == {
            "group": "aiperf.nvidia.com",
            "version": "v1alpha1",
            "namespace": "tenant-a",
            "plural": "aiperfsweeps",
            "body": cr,
        }
        mock_save.assert_called_once_with(
            "latency-sweep",
            "tenant-a",
            name=None,
            kind="AIPerfSweep",
        )

    @pytest.mark.asyncio
    async def test_submit_sweep_conflict_error_names_namespace_and_sweep(self) -> None:
        from aiperf.cli_commands.kube.sweep import _submit_sweep

        custom = MagicMock()
        custom.create_namespaced_custom_object = AsyncMock(
            side_effect=_api_exception(409)
        )
        cr = {
            "metadata": {"name": "latency-sweep"},
            "spec": {"image": "aiperf:branch"},
        }
        with (
            patch("aiperf.kubernetes.client.k8s_client", new=_fake_k8s_client),
            patch("kubernetes_asyncio.client.CustomObjectsApi", return_value=custom),
            pytest.raises(
                RuntimeError, match=r"tenant-a/latency-sweep.*already exists"
            ),
        ):
            await _submit_sweep(
                cr_dict=cr,
                kube_options=KubeOptions(image="aiperf:branch"),
                namespace="tenant-a",
            )

    @pytest.mark.asyncio
    async def test_sweep_dry_run_json_output_is_parseable_and_skips_submit(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from aiperf.cli_commands.kube.sweep import sweep

        cr = {
            "apiVersion": "aiperf.nvidia.com/v1alpha1",
            "kind": "AIPerfSweep",
            "metadata": {"name": "latency-sweep"},
            "spec": {"image": "aiperf:branch"},
        }
        with (
            patch(
                "aiperf.cli_commands.kube.sweep._build_sweep_cr_dict",
                return_value=cr,
            ),
            patch(
                "aiperf.cli_commands.kube.sweep._submit_sweep",
                new=AsyncMock(),
            ) as mock_submit,
        ):
            await sweep(
                cli_config=CLIConfig(config_file=Path("/configs/latency-sweep.yaml")),
                kube_options=KubeOptions(
                    image="aiperf:branch",
                    namespace="tenant-a",
                ),
                dry_run=True,
            )

        payload = orjson.loads(capsys.readouterr().out)
        assert payload == cr
        assert payload["metadata"]["namespace"] == "tenant-a"
        mock_submit.assert_not_awaited()

    @pytest.mark.parametrize(
        "output",
        ["text", "json"],
    )  # fmt: skip
    def test_render_list_runs_payload_honors_output_without_cross_talk(
        self, output: Literal["text", "json"], capsys: pytest.CaptureFixture[str]
    ) -> None:
        from aiperf.cli_commands.kube.results import _render_list_runs_payload

        payload = {
            "namespace": "bench-prod",
            "job_id": "latency-sweep-v03-t0",
            "runs": [{"run": "1770001234"}],
        }
        with patch(
            "aiperf.cli_commands.kube._runs_render.print_runs_table"
        ) as mock_table:
            _render_list_runs_payload(payload, output=output, preview=False)

        captured = capsys.readouterr().out
        if output == "json":
            assert orjson.loads(captured) == payload
            mock_table.assert_not_called()
        else:
            assert captured == ""
            mock_table.assert_called_once_with(payload, preview=False)

    @pytest.mark.asyncio
    async def test_run_sweep_results_passes_kube_context_to_operator_download(
        self, tmp_path: Path
    ) -> None:
        from aiperf.cli_commands.kube.results import _run_sweep_results

        resolved = _ResolvedSweepStub()
        opts = KubeManageOptions(
            namespace="tenant-a",
            kubeconfig="/secure/kubeconfigs/dgx-prod.yaml",
            kube_context="dgx-prod-admin",
        )
        with (
            patch(
                "aiperf.cli_commands.kube.results._resolve_op_ns",
                new=AsyncMock(return_value="aiperf-ops"),
            ),
            patch(
                "aiperf.kubernetes.results.retrieve_sweep_results_from_operator",
                new=AsyncMock(return_value=True),
            ) as mock_retrieve,
            patch("aiperf.kubernetes.console.print_results_summary"),
        ):
            await _run_sweep_results(
                resolved=resolved,
                output=tmp_path / "latency-sweep-results",
                from_pods=False,
                run="1770001234",
                manage_options=opts,
                operator_namespace="aiperf-ops",
                port=19091,
            )

        assert resolved.api.closed is True
        assert mock_retrieve.await_args.args[:3] == (
            "latency-sweep",
            "bench-prod",
            tmp_path / "latency-sweep-results",
        )
        assert mock_retrieve.await_args.kwargs == {
            "local_port": 19091,
            "operator_namespace": "aiperf-ops",
            "kubeconfig": "/secure/kubeconfigs/dgx-prod.yaml",
            "kube_context": "dgx-prod-admin",
            "run": "1770001234",
        }


# ============================================================
# URL and path encoding
# ============================================================


class TestSweepArtifactEncoding:
    """Operator-provided artifact names must not escape paths or break URLs."""

    @pytest.mark.asyncio
    async def test_download_sweep_operator_file_url_quotes_special_path_segments(
        self, tmp_path: Path
    ) -> None:
        from aiperf.kubernetes.results_operator import _download_sweep_operator_file

        session = _Session(_Response(body=b"profile"))
        result = await _download_sweep_operator_file(
            session,
            api_base="http://operator",
            namespace="bench-prod",
            sweep_name="latency-sweep",
            run="1770001234",
            file_info={"name": "sweep_aggregate/profile export #1.json"},
            output_dir=tmp_path,
        )

        assert result == ("sweep_aggregate/profile export #1.json", len(b"profile"))
        assert session.requested == [
            (
                "http://operator/api/v1/sweeps/bench-prod/latency-sweep/epochs/1770001234/artifacts/"
                "sweep_aggregate/profile%20export%20%231.json",
                {"Accept-Encoding": "zstd, gzip, identity"},
            )
        ]
        assert (
            tmp_path / "sweep_aggregate" / "profile export #1.json"
        ).read_bytes() == b"profile"

    @pytest.mark.parametrize(
        "artifact_name",
        [
            "../profile_export_aiperf.json",
            "sweep_aggregate/../metrics.json",
            ".hidden-profile.json",
            "sweep_aggregate/.hidden-profile.json",
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_download_sweep_operator_file_refuses_unsafe_paths(
        self, tmp_path: Path, artifact_name: str
    ) -> None:
        from aiperf.kubernetes.results_operator import _download_sweep_operator_file

        session = _Session(_Response(body=b"should-not-download"))
        with patch(
            "aiperf.kubernetes.results_operator_sweeps.print_warning"
        ) as mock_warning:
            result = await _download_sweep_operator_file(
                session,
                api_base="http://operator",
                namespace="bench-prod",
                sweep_name="latency-sweep",
                run="1770001234",
                file_info={"name": artifact_name},
                output_dir=tmp_path,
            )

        assert result is None
        assert session.requested == []
        assert "unsafe" in mock_warning.call_args.args[0]
