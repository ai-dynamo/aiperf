# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for AIPerfSweep aggregate harvesting.

Focuses on:
- no-PVC sidecar harvest before sweep-controller JobSet cleanup;
- transient-vs-programmer error surfaces from the operator-side sidecar fetch;
- malformed and missing aggregate bundle files loaded by the sweep-controller;
- path-traversal-shaped aggregate artifacts that must not escape the mirror dir;
- latest-pointer timing around partial or failed aggregate writes.

Out of scope: full kopf registration and UI rendering; see sibling operator main,
sweep-controller main, and sweep-detail UI tests for those surfaces.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import kopf
import orjson
import pytest
from pytest import param

from aiperf.operator import main as operator_main
from aiperf.operator.environment import OperatorEnvironment
from aiperf.operator.handlers.sweep import _aggregate_fetch as aggregate_fetch
from aiperf.sweep_controller import main as sweep_main
from aiperf.sweep_controller.k8s_executor import ChildRunRef

# ============================================================
# Helpers
# ============================================================


def _write_json(path: Path, doc: object) -> None:
    """Write an aggregate fixture at the production path shape."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(doc))


def _sweep_epoch_dir(base_dir: Path) -> Path:
    """Return the canonical parent aggregate epoch dir for the latency sweep."""
    return base_dir / "aiperf-benchmarks" / "sweeps" / "latency-sweep" / "1778027124"


def _fake_progress_client(
    downloaded: list[str] | BaseException,
    *,
    listed: list[str] | None = None,
) -> MagicMock:
    """Build a ProgressClient double that records sidecar list+download calls.

    ``listed`` defaults to mirroring ``downloaded`` (a full harvest); pass it
    explicitly to simulate a partial harvest. When ``downloaded`` is an
    exception, the listing raises it too (transport failures hit the first
    HTTP call).
    """
    fake = MagicMock()
    fake.__aenter__ = AsyncMock(return_value=fake)
    fake.__aexit__ = AsyncMock(return_value=None)
    if isinstance(downloaded, BaseException):
        fake.get_results_list = AsyncMock(side_effect=downloaded)
        fake.download_all_results = AsyncMock(side_effect=downloaded)
    else:
        names = listed if listed is not None else downloaded
        fake.get_results_list = AsyncMock(
            return_value=[{"name": name, "size": 64} for name in names]
        )
        fake.download_all_results = AsyncMock(return_value=downloaded)
    return fake


def _one_successful_child_result() -> SimpleNamespace:
    """Build a minimal successful RunResult-shaped object for parent aggregate writes."""
    return SimpleNamespace(
        label="latency-sweep-v00-t0",
        success=True,
        error=None,
        variation_values={"index": 0},
        variation_index=0,
        variation_label="concurrency=64",
        trial_index=0,
        child_run_epoch="1778027999",
    )


# ============================================================
# Operator-side sidecar harvest
# ============================================================


class TestSweepAggregateSidecarFetch:
    """No-PVC harvest contracts for the sweep-controller results sidecar."""

    @pytest.mark.asyncio
    async def test_fetch_sweep_aggregate_uses_controller_sidecar_and_operator_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_client = _fake_progress_client(
            [
                "aiperf-benchmarks/sweeps/latency-sweep/1778027124/aggregate.json",
                "aiperf-benchmarks/sweeps/latency-sweep/1778027124/children.json",
            ]
        )
        constructed_ports: list[int | None] = []

        def fake_progress_client(*, port: int | None = None) -> MagicMock:
            constructed_ports.append(port)
            return fake_client

        monkeypatch.setattr(aggregate_fetch, "ProgressClient", fake_progress_client)

        result = await aggregate_fetch.fetch_sweep_aggregate_to_disk(
            sweep_name="latency-sweep",
            namespace="aiperf-benchmarks",
            epoch="1778027124",
            base_dir=tmp_path,
        )

        assert result.downloaded == 2
        assert result.listed == 2
        assert constructed_ports == [
            aggregate_fetch.K8sEnvironment.PORTS.RESULTS_SIDECAR
        ]
        host_arg, dest_arg = fake_client.download_all_results.await_args.args
        assert host_arg == (
            "aiperf-latency-sweep-controller-0-0."
            "aiperf-latency-sweep.aiperf-benchmarks.svc.cluster.local"
        )
        assert dest_arg == tmp_path
        assert not (
            tmp_path / "aiperf-benchmarks" / "sweeps" / "latency-sweep" / "latest.txt"
        ).exists()

    @pytest.mark.asyncio
    async def test_fetch_sweep_aggregate_partial_download_reports_listed_gap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A partial harvest reports the gap and never advances latest."""
        fake_client = _fake_progress_client(
            ["aiperf-benchmarks/sweeps/latency-sweep/1778027124/aggregate.json"],
            listed=[
                "aiperf-benchmarks/sweeps/latency-sweep/1778027124/aggregate.json",
                "aiperf-benchmarks/sweeps/latency-sweep/1778027124/children.json",
            ],
        )
        monkeypatch.setattr(
            aggregate_fetch, "ProgressClient", lambda *args, **kwargs: fake_client
        )

        result = await aggregate_fetch.fetch_sweep_aggregate_to_disk(
            sweep_name="latency-sweep",
            namespace="aiperf-benchmarks",
            epoch="1778027124",
            base_dir=tmp_path,
        )

        assert result.downloaded == 1
        assert result.listed == 2
        assert result.is_partial
        assert not (
            tmp_path / "aiperf-benchmarks" / "sweeps" / "latency-sweep" / "latest.txt"
        ).exists()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [
            param(TimeoutError("sidecar read timed out"), id="timeout"),
            param(ConnectionError("controller pod already gone"), id="connection-error"),
            param(aiohttp.ClientConnectionError("dns not ready"), id="aiohttp-client-error"),
        ],
    )  # fmt: skip
    async def test_fetch_sweep_aggregate_transient_errors_return_zero_without_latest(
        self,
        error: BaseException,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        fake_client = _fake_progress_client(error)
        monkeypatch.setattr(
            aggregate_fetch, "ProgressClient", lambda *args, **kwargs: fake_client
        )

        with caplog.at_level("WARNING", logger=aggregate_fetch.logger.name):
            result = await aggregate_fetch.fetch_sweep_aggregate_to_disk(
                sweep_name="latency-sweep",
                namespace="aiperf-benchmarks",
                epoch="1778027124",
                base_dir=tmp_path,
            )

        assert result.downloaded == 0
        assert result.listed == 0
        assert not (
            tmp_path / "aiperf-benchmarks" / "sweeps" / "latency-sweep" / "latest.txt"
        ).exists()
        assert "aiperf-benchmarks/latency-sweep" in caplog.text
        assert str(aggregate_fetch.K8sEnvironment.PORTS.RESULTS_SIDECAR) in caplog.text

    @pytest.mark.asyncio
    async def test_fetch_sweep_aggregate_unexpected_error_propagates_without_latest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_client = _fake_progress_client(
            RuntimeError("download manifest shape changed unexpectedly")
        )
        monkeypatch.setattr(
            aggregate_fetch, "ProgressClient", lambda *args, **kwargs: fake_client
        )

        with pytest.raises(RuntimeError, match="download manifest shape changed"):
            await aggregate_fetch.fetch_sweep_aggregate_to_disk(
                sweep_name="latency-sweep",
                namespace="aiperf-benchmarks",
                epoch="1778027124",
                base_dir=tmp_path,
            )

        assert not (
            tmp_path / "aiperf-benchmarks" / "sweeps" / "latency-sweep" / "latest.txt"
        ).exists()

    @pytest.mark.asyncio
    async def test_on_aiperfsweep_aggregation_complete_zero_fetch_surfaces_retry_reason(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fetch = AsyncMock(
            return_value=aggregate_fetch.SweepAggregateFetchResult(
                downloaded=0, listed=0
            )
        )
        monkeypatch.setattr(
            aggregate_fetch,
            "fetch_sweep_aggregate_to_disk",
            fetch,
        )
        monkeypatch.setattr(
            operator_main,
            "_sweep_parent_is_current",
            AsyncMock(return_value=True),
        )
        monkeypatch.setattr(
            operator_main,
            "_owned_sweep_jobset_uid",
            AsyncMock(return_value="jobset-uid"),
        )
        monkeypatch.setattr(OperatorEnvironment.RESULTS, "DIR", tmp_path)

        with pytest.raises(
            kopf.TemporaryError,
            match=r"aiperf-benchmarks/latency-sweep.*sidecar returned no files",
        ):
            await operator_main.on_aiperfsweep_aggregation_complete(
                body={"metadata": {"uid": "sweep-uid"}},
                status={
                    "runEpoch": "1778027124",
                    "aggregation": {"phase": "Complete"},
                },
                name="latency-sweep",
                namespace="aiperf-benchmarks",
            )

        fetch.assert_awaited_once_with(
            sweep_name="latency-sweep",
            namespace="aiperf-benchmarks",
            epoch="1778027124",
            base_dir=tmp_path,
        )


# ============================================================
# Sweep-controller aggregate bundle loading
# ============================================================


class TestSweepControllerAggregateBundleLoading:
    """Malformed bundle artifacts degrade to partial CR mirrors, not pod crashes."""

    @pytest.mark.parametrize(
        "bad_key,bad_path_parts,expected_keys",
        [
            param(
                "parent",
                ("aiperf-benchmarks", "sweeps", "latency-sweep", "1778027124", "aggregate.json"),
                {"children", "confidence"},
                id="malformed-parent-keeps-children-confidence",
            ),
            param(
                "children",
                ("aiperf-benchmarks", "sweeps", "latency-sweep", "1778027124", "children.json"),
                {"parent", "confidence"},
                id="malformed-children-keeps-parent-confidence",
            ),
            param(
                "confidence",
                ("aggregate", "profile_export_aiperf_aggregate.json"),
                {"parent", "children"},
                id="malformed-confidence-keeps-parent-children",
            ),
        ],
    )  # fmt: skip
    def test_load_aggregate_for_cr_malformed_file_skips_only_bad_artifact(
        self,
        bad_key: str,
        bad_path_parts: tuple[str, ...],
        expected_keys: set[str],
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        sweep_dir = _sweep_epoch_dir(tmp_path)
        _write_json(sweep_dir / "aggregate.json", {"phase": "Succeeded"})
        _write_json(
            sweep_dir / "children.json",
            {"children": [{"name": "latency-sweep-v00-t0", "status": "Succeeded"}]},
        )
        _write_json(
            tmp_path / "aggregate" / "profile_export_aiperf_aggregate.json",
            {"summary": {"output_token_throughput": 2048.0}},
        )
        bad_path = tmp_path.joinpath(*bad_path_parts)
        bad_path.write_bytes(b'{"truncated":')

        with caplog.at_level("WARNING", logger=sweep_main.logger.name):
            bundle = sweep_main._load_aggregate_for_cr(
                tmp_path,
                "aiperf-benchmarks",
                "latency-sweep",
                "1778027124",
            )

        assert set(bundle) == expected_keys
        assert f"skipping {bad_key}" in caplog.text
        assert str(bad_path) in caplog.text

    def test_load_aggregate_for_cr_missing_all_files_returns_empty_bundle(
        self, tmp_path: Path
    ) -> None:
        bundle = sweep_main._load_aggregate_for_cr(
            tmp_path,
            "aiperf-benchmarks",
            "latency-sweep",
            "1778027124",
        )

        assert bundle == {}

    def test_load_aggregate_for_cr_non_json_bytes_are_skipped_with_context(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        sweep_dir = _sweep_epoch_dir(tmp_path)
        _write_json(sweep_dir / "aggregate.json", {"phase": "Succeeded"})
        (tmp_path / "aggregate").mkdir(parents=True)
        confidence_path = (
            tmp_path / "aggregate" / "profile_export_aiperf_aggregate.json"
        )
        confidence_path.write_bytes(b"not-json-from-sidecar")

        with caplog.at_level("WARNING", logger=sweep_main.logger.name):
            bundle = sweep_main._load_aggregate_for_cr(
                tmp_path,
                "aiperf-benchmarks",
                "latency-sweep",
                "1778027124",
            )

        assert bundle == {"parent": {"phase": "Succeeded"}}
        assert "skipping confidence" in caplog.text
        assert str(confidence_path) in caplog.text
        assert "JSONDecodeError" in caplog.text


# ============================================================
# Aggregate artifact path traversal and latest-pointer timing
# ============================================================


class TestAggregateArtifactTrustBoundaries:
    """Filesystem trust boundaries for aggregate artifact mirroring and latest pointers."""

    def test_mirror_strategy_aggregate_to_sweep_dir_skips_symlink_escape(
        self, tmp_path: Path
    ) -> None:
        aggregate_dir = tmp_path / "aggregate"
        aggregate_dir.mkdir()
        outside_secret = tmp_path / "operator-token.txt"
        outside_secret.write_text("cluster-admin-token")
        (aggregate_dir / "profile_export_aiperf_aggregate.json").write_text(
            '{"summary":"ok"}'
        )
        (aggregate_dir / "escaped-token.json").symlink_to(outside_secret)

        sweep_main._mirror_strategy_aggregate_to_sweep_dir(
            base_dir=tmp_path,
            aggregate_dir=aggregate_dir,
            namespace="aiperf-benchmarks",
            sweep_name="latency-sweep",
            sweep_run_epoch="1778027124",
        )

        target = _sweep_epoch_dir(tmp_path) / "sweep_aggregate"
        assert sorted(p.name for p in target.iterdir()) == [
            "profile_export_aiperf_aggregate.json"
        ]
        assert not (target / "escaped-token.json").exists()

    def test_write_sweep_parent_aggregate_children_failure_does_not_advance_latest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail_children_manifest(**_kwargs: object) -> None:
            raise OSError("children manifest PVC write failed")

        import aiperf.sweep_controller.aggregator as aggregator

        monkeypatch.setattr(
            aggregator, "write_children_manifest", fail_children_manifest
        )

        with pytest.raises(OSError, match="children manifest PVC write failed"):
            sweep_main._write_sweep_parent_aggregate(
                base_dir=tmp_path,
                sweep_cr={
                    "metadata": {
                        "namespace": "aiperf-benchmarks",
                        "name": "latency-sweep",
                    }
                },
                spec=SimpleNamespace(
                    model_dump=lambda mode, by_alias: {"benchmark": {}}
                ),
                results=[_one_successful_child_result()],
                child_runs=[
                    ChildRunRef(
                        namespace="aiperf-benchmarks",
                        name="latency-sweep-v00-t0",
                        variation_index=0,
                        variation_label="concurrency=64",
                        trial_index=0,
                        child_run_epoch="1778027999",
                        label="latency-sweep-v00-t0",
                        status="Succeeded",
                        error="",
                    )
                ],
                plan=SimpleNamespace(configs=[object()]),
                sweep_run_epoch="1778027124",
                terminal_phase="Succeeded",
            )

        latest = (
            tmp_path / "aiperf-benchmarks" / "sweeps" / "latency-sweep" / "latest.txt"
        )
        assert not latest.exists()
