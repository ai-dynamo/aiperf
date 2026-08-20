# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``aiperf.operator.handlers._completion_fetch``.

Covers the low-level result-fetch retry/backoff machinery:

- ``_snapshot_bytes`` — directory size accounting with vanishing-file tolerance.
- ``_split_downloaded`` — checkpoints vs final-file partitioning.
- ``_merge_downloaded`` — sorted-set union semantics across retry attempts.
- ``_IncompleteResultsError.to_fetch_result`` — error-message phrasing per
  partial-progress shape.
- ``_update_progress_streak`` — stagnation counter + raise-on-stall contract.
- ``_try_fetch_once`` — single-attempt exception-to-tuple translation.
- ``_fetch_with_progress_aware_retry`` — progress-aware retry loop, including
  the cancellation short-circuit branches.
- ``_run_fetch_loop_safely`` — failure-to-ControllerFetchResult coercion.
- ``_download_final_and_sidecar`` — primary-then-sidecar download orchestration
  with key-file / port-collision short-circuits.
- ``_fetch_once_into_state`` — single fetch attempt that mutates shared state.
- ``fetch_results_with_retry`` — public entry: ValueError on missing inputs,
  dest_dir derivation from body, cancellation handling.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from kubernetes_asyncio.client.exceptions import ApiException
from pytest import param

from aiperf.common.results_markers import CHECKPOINTS_DIR_NAME
from aiperf.kubernetes.crd_models import ControllerFetchResult
from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.operator.handlers import _completion_fetch as cf
from aiperf.operator.handlers._completion_fetch import (
    _download_final_and_sidecar,
    _fetch_once_into_state,
    _fetch_with_progress_aware_retry,
    _IncompleteResultsError,
    _merge_downloaded,
    _run_fetch_loop_safely,
    _snapshot_bytes,
    _split_downloaded,
    _try_fetch_once,
    _update_progress_streak,
    fetch_results_with_retry,
)

# =====================================================================
# Helpers / fixtures
# =====================================================================


def _success_result(
    metrics: dict[str, Any] | None = None,
    downloaded: list[str] | None = None,
    checkpoints: list[str] | None = None,
) -> ControllerFetchResult:
    return ControllerFetchResult(
        metrics=metrics or {"some": "metric"},
        downloaded=downloaded or ["profile_export_aiperf.json"],
        checkpoints=checkpoints or [],
    )


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace asyncio.sleep inside _completion_fetch with an instant async no-op."""
    monkeypatch.setattr(cf.asyncio, "sleep", AsyncMock())


@pytest.fixture(autouse=True)
def _deterministic_jitter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin random.uniform so backoff is deterministic in tests."""
    monkeypatch.setattr(cf.random, "uniform", lambda _a, _b: 1.0)


# =====================================================================
# _snapshot_bytes
# =====================================================================


class TestSnapshotBytes:
    def test_nonexistent_dir_returns_zero(self, tmp_path: Path) -> None:
        assert _snapshot_bytes(tmp_path / "missing") == 0

    def test_empty_dir_returns_zero(self, tmp_path: Path) -> None:
        assert _snapshot_bytes(tmp_path) == 0

    def test_single_file_returns_size(self, tmp_path: Path) -> None:
        f = tmp_path / "a.txt"
        f.write_bytes(b"hello")
        assert _snapshot_bytes(tmp_path) == 5

    def test_nested_files_sums_all_sizes(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_bytes(b"abc")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.txt").write_bytes(b"defgh")
        deeper = sub / "deeper"
        deeper.mkdir()
        (deeper / "c.bin").write_bytes(b"ZZ")
        assert _snapshot_bytes(tmp_path) == 3 + 5 + 2

    def test_vanishing_file_is_silently_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        good = tmp_path / "good.txt"
        good.write_bytes(b"1234")
        ghost = tmp_path / "ghost.txt"
        ghost.write_bytes(b"will-vanish")

        real_stat = Path.stat
        # is_file() calls stat() internally and falls back to False on OSError;
        # we only want the *second* stat (the explicit st_size lookup at line 94)
        # to raise, so let is_file succeed and only fail the size read.
        ghost_calls = {"n": 0}

        def fake_stat(self: Path, *args: Any, **kwargs: Any) -> Any:
            if self.name == "ghost.txt":
                ghost_calls["n"] += 1
                if ghost_calls["n"] >= 2:
                    raise OSError("vanished")
            return real_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", fake_stat)
        assert _snapshot_bytes(tmp_path) == 4


# =====================================================================
# _split_downloaded
# =====================================================================


class TestSplitDownloaded:
    @pytest.mark.parametrize(
        "paths,expected_final,expected_ckpt",
        [
            (None, [], []),
            ([], [], []),
            param(
                [f"{CHECKPOINTS_DIR_NAME}/c1.json", "profile_export_aiperf.json"],
                ["profile_export_aiperf.json"],
                [f"{CHECKPOINTS_DIR_NAME}/c1.json"],
                id="mixed-checkpoint-and-final",
            ),
            param(
                [f"{CHECKPOINTS_DIR_NAME}/c1.json", f"{CHECKPOINTS_DIR_NAME}/c2.json"],
                [],
                [f"{CHECKPOINTS_DIR_NAME}/c1.json", f"{CHECKPOINTS_DIR_NAME}/c2.json"],
                id="all-checkpoints",
            ),
            param(
                ["a.json", "b.csv"],
                ["a.json", "b.csv"],
                [],
                id="all-final-files",
            ),
        ],
    )  # fmt: skip
    def test_split_downloaded_partitions_correctly(
        self,
        paths: list[str] | None,
        expected_final: list[str],
        expected_ckpt: list[str],
    ) -> None:
        final, ckpt = _split_downloaded(paths)
        assert final == expected_final
        assert ckpt == expected_ckpt

    def test_split_uses_checkpoints_dir_constant_prefix(self) -> None:
        """A path starting with ``{CHECKPOINTS_DIR_NAME}/`` is a checkpoint."""
        prefix = f"{CHECKPOINTS_DIR_NAME}/foo.json"
        final, ckpt = _split_downloaded([prefix])
        assert final == []
        assert ckpt == [prefix]


# =====================================================================
# _merge_downloaded
# =====================================================================


class TestMergeDownloaded:
    def test_new_none_returns_current_unchanged(self) -> None:
        current = ["a", "b"]
        assert _merge_downloaded(current, None) is current

    def test_current_none_returns_list_copy_of_new(self) -> None:
        new = ["x", "y"]
        result = _merge_downloaded(None, new)
        assert result == ["x", "y"]
        # Should be a new list (so future mutations of `new` don't bleed in).
        assert result is not new

    def test_both_populated_unions_and_sorts(self) -> None:
        assert _merge_downloaded(["b", "c"], ["a", "d"]) == ["a", "b", "c", "d"]

    def test_overlap_is_de_duplicated(self) -> None:
        assert _merge_downloaded(["a", "b"], ["b", "c"]) == ["a", "b", "c"]

    def test_both_none_returns_none(self) -> None:
        assert _merge_downloaded(None, None) is None


# =====================================================================
# _IncompleteResultsError.to_fetch_result
# =====================================================================


class TestIncompleteResultsErrorToFetchResult:
    def test_no_metrics_no_downloaded_uses_failed_to_fetch_results(self) -> None:
        err = _IncompleteResultsError(metrics=None, downloaded=[], checkpoints=[])
        out = err.to_fetch_result("ns/job")
        assert out.error == "Failed to fetch results"
        assert out.metrics is None
        assert out.downloaded == []
        assert out.checkpoints == []

    def test_no_metrics_has_downloaded(self) -> None:
        err = _IncompleteResultsError(
            metrics=None, downloaded=["a.json"], checkpoints=["checkpoints/c1"]
        )
        out = err.to_fetch_result("ns/job")
        assert out.error == "Failed to fetch metrics (files downloaded)"
        assert out.downloaded == ["a.json"]
        assert out.checkpoints == ["checkpoints/c1"]

    def test_has_metrics_no_downloaded(self) -> None:
        err = _IncompleteResultsError(metrics={"k": "v"}, downloaded=[], checkpoints=[])
        out = err.to_fetch_result("ns/job")
        assert out.error == "Failed to download result files (metrics fetched)"
        assert out.metrics == {"k": "v"}

    def test_has_metrics_and_downloaded_yields_empty_error(self) -> None:
        err = _IncompleteResultsError(
            metrics={"k": "v"}, downloaded=["a.json"], checkpoints=["checkpoints/c1"]
        )
        out = err.to_fetch_result("ns/job")
        assert out.error == ""
        assert out.metrics == {"k": "v"}
        assert out.downloaded == ["a.json"]
        assert out.checkpoints == ["checkpoints/c1"]


# =====================================================================
# _update_progress_streak
# =====================================================================


class TestUpdateProgressStreak:
    def test_progress_resets_streak_to_zero(self) -> None:
        assert (
            _update_progress_streak(
                bytes_now=200,
                last_bytes=100,
                streak=3,
                stagnation_limit=5,
                description="d",
                attempt=1,
                delay=1.0,
                pending_exc=RuntimeError("pending"),
            )
            == 0
        )

    def test_no_progress_increments_streak(self) -> None:
        assert (
            _update_progress_streak(
                bytes_now=100,
                last_bytes=100,
                streak=2,
                stagnation_limit=5,
                description="d",
                attempt=1,
                delay=1.0,
                pending_exc=RuntimeError("pending"),
            )
            == 3
        )

    def test_bytes_decreased_still_no_progress(self) -> None:
        # Code only treats `bytes_now > last_bytes` as progress; equality OR
        # decrease both advance the streak.
        assert (
            _update_progress_streak(
                bytes_now=80,
                last_bytes=100,
                streak=0,
                stagnation_limit=5,
                description="d",
                attempt=1,
                delay=1.0,
                pending_exc=RuntimeError("pending"),
            )
            == 1
        )

    def test_stall_at_limit_raises_pending_exc(self) -> None:
        boom = RuntimeError("transient")
        with pytest.raises(RuntimeError, match="transient") as exc_info:
            _update_progress_streak(
                bytes_now=100,
                last_bytes=100,
                streak=4,
                stagnation_limit=5,
                description="d",
                attempt=10,
                delay=1.0,
                pending_exc=boom,
            )
        assert exc_info.value is boom

    def test_stall_with_none_pending_exc_violates_assertion(self) -> None:
        # Contract: pending_exc must not be None when stagnation triggers.
        with pytest.raises(AssertionError):
            _update_progress_streak(
                bytes_now=100,
                last_bytes=100,
                streak=4,
                stagnation_limit=5,
                description="d",
                attempt=10,
                delay=1.0,
                pending_exc=None,
            )


# =====================================================================
# _try_fetch_once
# =====================================================================


class TestTryFetchOnce:
    @pytest.mark.asyncio
    async def test_success_returns_result_and_no_exc(self) -> None:
        expected = _success_result()

        async def fetch() -> ControllerFetchResult:
            return expected

        result, exc = await _try_fetch_once(fetch, description="d", attempt=1)
        assert result is expected
        assert exc is None

    @pytest.mark.asyncio
    async def test_incomplete_results_error_returned_as_exc(self) -> None:
        boom = _IncompleteResultsError(metrics=None, downloaded=[], checkpoints=[])

        async def fetch() -> ControllerFetchResult:
            raise boom

        result, exc = await _try_fetch_once(fetch, description="d", attempt=1)
        assert result is None
        assert exc is boom

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc",
        [
            param(aiohttp.ClientError("boom"), id="aiohttp-client-error"),
            param(TimeoutError("slow"), id="asyncio-timeout"),
            param(OSError("disk-full"), id="os-error"),
            param(ApiException("k8s-api"), id="k8s-api-exception"),
        ],
    )  # fmt: skip
    async def test_known_transient_errors_returned_as_exc(
        self, exc: BaseException
    ) -> None:
        async def fetch() -> ControllerFetchResult:
            raise exc

        result, got_exc = await _try_fetch_once(fetch, description="d", attempt=1)
        assert result is None
        assert got_exc is exc

    @pytest.mark.asyncio
    async def test_arbitrary_exception_returned_as_exc(self) -> None:
        boom = ValueError("unexpected")

        async def fetch() -> ControllerFetchResult:
            raise boom

        result, exc = await _try_fetch_once(fetch, description="d", attempt=1)
        assert result is None
        assert exc is boom


# =====================================================================
# _fetch_with_progress_aware_retry
# =====================================================================


class TestFetchWithProgressAwareRetry:
    @pytest.mark.asyncio
    async def test_succeeds_on_first_attempt(self, tmp_path: Path) -> None:
        expected = _success_result()
        fetch_once = AsyncMock(return_value=expected)

        result = await _fetch_with_progress_aware_retry(
            fetch_once,
            dest_dir=tmp_path,
            job_id="job",
            initial_delay=0.1,
            description="d",
        )
        assert result is expected
        assert fetch_once.await_count == 1

    @pytest.mark.asyncio
    async def test_succeeds_after_transient_failures_with_progress(
        self, tmp_path: Path
    ) -> None:
        # Two failures with disk growing each time, then success.
        success = _success_result()
        attempts = {"n": 0}

        async def fetch_once() -> ControllerFetchResult:
            attempts["n"] += 1
            if attempts["n"] == 1:
                (tmp_path / "a.bin").write_bytes(b"x" * 10)
                raise _IncompleteResultsError(None, [], [])
            if attempts["n"] == 2:
                (tmp_path / "b.bin").write_bytes(b"y" * 20)
                raise _IncompleteResultsError({"k": "v"}, [], [])
            return success

        result = await _fetch_with_progress_aware_retry(
            fetch_once,
            dest_dir=tmp_path,
            job_id="job",
            initial_delay=0.1,
            description="d",
        )
        assert result is success
        assert attempts["n"] == 3

    @pytest.mark.asyncio
    async def test_stagnation_limit_raises_last_pending(self, tmp_path: Path) -> None:
        # Disk never grows; bubble the most recent exception after limit hits.
        boom = _IncompleteResultsError(None, [], [])

        async def fetch_once() -> ControllerFetchResult:
            raise boom

        with pytest.raises(_IncompleteResultsError):
            await _fetch_with_progress_aware_retry(
                fetch_once,
                dest_dir=tmp_path,
                job_id="job",
                initial_delay=0.1,
                description="d",
                stagnation_limit=3,
            )

    @pytest.mark.asyncio
    async def test_cancellation_before_first_attempt_short_circuits(
        self, tmp_path: Path
    ) -> None:
        # is_cancelled True at top of loop -> calls fetch_once exactly once and
        # returns whatever it produced (even errors propagate).
        expected = _success_result()
        fetch_once = AsyncMock(return_value=expected)

        result = await _fetch_with_progress_aware_retry(
            fetch_once,
            dest_dir=tmp_path,
            job_id="job",
            initial_delay=0.1,
            description="d",
            is_cancelled=lambda: True,
        )
        assert result is expected
        assert fetch_once.await_count == 1

    @pytest.mark.asyncio
    async def test_cancellation_between_attempts_short_circuits(
        self, tmp_path: Path
    ) -> None:
        """Mid-loop is_cancelled flip routes through the post-attempt
        cancellation branch (line `if is_cancelled is not None and is_cancelled():`
        AFTER _try_fetch_once)."""
        success = _success_result()
        cancel_state = {"flag": False}
        fetch_once = AsyncMock(
            side_effect=[
                _IncompleteResultsError(None, [], []),
                success,
            ]
        )

        def is_cancelled() -> bool:
            # First attempt: not cancelled. After first failure we flip True so
            # the post-attempt branch trips and re-calls fetch_once.
            return cancel_state["flag"]

        # Toggle cancel after the first fetch_once raises.
        original_side = fetch_once.side_effect

        async def wrapper() -> ControllerFetchResult:
            try:
                return await AsyncMock(side_effect=original_side)()
            finally:
                cancel_state["flag"] = True

        # Simpler: flip the flag inside fetch_once itself.
        attempts = {"n": 0}

        async def fetch_once_real() -> ControllerFetchResult:
            attempts["n"] += 1
            if attempts["n"] == 1:
                cancel_state["flag"] = True
                raise _IncompleteResultsError(None, [], [])
            return success

        result = await _fetch_with_progress_aware_retry(
            fetch_once_real,
            dest_dir=tmp_path,
            job_id="job",
            initial_delay=0.1,
            description="d",
            is_cancelled=is_cancelled,
        )
        assert result is success
        assert attempts["n"] == 2

    @pytest.mark.asyncio
    async def test_backoff_uses_asyncio_sleep_between_attempts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sleep_mock = AsyncMock()
        monkeypatch.setattr(cf.asyncio, "sleep", sleep_mock)

        success = _success_result()
        attempts = {"n": 0}

        async def fetch_once() -> ControllerFetchResult:
            attempts["n"] += 1
            if attempts["n"] < 3:
                (tmp_path / f"f{attempts['n']}.bin").write_bytes(b"x" * 10)
                raise _IncompleteResultsError(None, [], [])
            return success

        result = await _fetch_with_progress_aware_retry(
            fetch_once,
            dest_dir=tmp_path,
            job_id="job",
            initial_delay=1.0,
            description="d",
            max_delay=4.0,
            backoff_multiplier=2.0,
        )
        assert result is success
        # Two retries -> two sleeps; both with jitter == 1.0 (deterministic
        # fixture). Delays observed: 1.0 then capped after backoff to 2.0.
        assert sleep_mock.await_count == 2
        observed = [call.args[0] for call in sleep_mock.await_args_list]
        assert observed == [1.0, 2.0]

    @pytest.mark.asyncio
    async def test_cancellation_during_backoff_returns_partial_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cancellation_event = asyncio.Event()
        sleep_started = asyncio.Event()
        never = asyncio.Event()

        async def blocking_sleep(_delay: float) -> None:
            sleep_started.set()
            await never.wait()

        async def fetch_once() -> ControllerFetchResult:
            raise _IncompleteResultsError({"kept": 1}, ["partial.json"], [])

        monkeypatch.setattr(cf.asyncio, "sleep", blocking_sleep)
        state = {
            "metrics": {"kept": 1},
            "downloaded": ["partial.json"],
            "checkpoints": None,
        }
        task = asyncio.create_task(
            _run_fetch_loop_safely(
                fetch_once,
                dest_dir=tmp_path,
                job_id="ns/job",
                retry_delay=30.0,
                stagnation_limit=5,
                is_cancelled=cancellation_event.is_set,
                state=state,
                cancellation_event=cancellation_event,
            )
        )

        await sleep_started.wait()
        cancellation_event.set()
        result = await task

        assert result.error == "Cancelled by CR deletion"
        assert result.metrics == {"kept": 1}
        assert result.downloaded == ["partial.json"]


# =====================================================================
# _run_fetch_loop_safely
# =====================================================================


class TestRunFetchLoopSafely:
    def _state(self, **overrides: Any) -> dict[str, Any]:
        s: dict[str, Any] = {
            "metrics": None,
            "downloaded": None,
            "checkpoints": None,
        }
        s.update(overrides)
        return s

    @pytest.mark.asyncio
    async def test_success_passes_result_through(self, tmp_path: Path) -> None:
        success = _success_result()
        fetch_once = AsyncMock(return_value=success)

        out = await _run_fetch_loop_safely(
            fetch_once,
            dest_dir=tmp_path,
            job_id="ns/job",
            retry_delay=0.1,
            stagnation_limit=3,
            is_cancelled=lambda: False,
            state=self._state(),
        )
        assert out is success

    @pytest.mark.asyncio
    async def test_incomplete_results_error_converted(self, tmp_path: Path) -> None:
        async def fetch_once() -> ControllerFetchResult:
            raise _IncompleteResultsError(
                metrics={"x": 1}, downloaded=[], checkpoints=[]
            )

        out = await _run_fetch_loop_safely(
            fetch_once,
            dest_dir=tmp_path,
            job_id="ns/job",
            retry_delay=0.1,
            stagnation_limit=2,
            is_cancelled=lambda: False,
            state=self._state(),
        )
        assert out.error == "Failed to download result files (metrics fetched)"
        assert out.metrics == {"x": 1}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "raised",
        [
            param(aiohttp.ClientError("conn-refused"), id="aiohttp"),
            param(TimeoutError("slow"), id="timeout"),
            param(OSError("disk"), id="os"),
            param(ApiException("k8s"), id="k8s-api"),
        ],
    )  # fmt: skip
    async def test_known_transient_returns_partial_state(
        self, tmp_path: Path, raised: BaseException
    ) -> None:
        async def fetch_once() -> ControllerFetchResult:
            raise raised

        partial = self._state(
            metrics={"k": "v"}, downloaded=["a.json"], checkpoints=["c1"]
        )
        out = await _run_fetch_loop_safely(
            fetch_once,
            dest_dir=tmp_path,
            job_id="ns/job",
            retry_delay=0.1,
            stagnation_limit=2,
            is_cancelled=lambda: False,
            state=partial,
        )
        assert out.metrics == {"k": "v"}
        assert out.downloaded == ["a.json"]
        assert out.checkpoints == ["c1"]
        assert out.error.startswith("Failed to fetch results: ")

    @pytest.mark.asyncio
    async def test_arbitrary_exception_returns_partial_state(
        self, tmp_path: Path
    ) -> None:
        async def fetch_once() -> ControllerFetchResult:
            raise ValueError("unexpected")

        out = await _run_fetch_loop_safely(
            fetch_once,
            dest_dir=tmp_path,
            job_id="ns/job",
            retry_delay=0.1,
            stagnation_limit=2,
            is_cancelled=lambda: False,
            state=self._state(),
        )
        assert out.error.startswith("Failed to fetch results: ")
        assert out.metrics is None
        assert out.downloaded == []
        assert out.checkpoints == []


# =====================================================================
# _download_final_and_sidecar
# =====================================================================


class TestDownloadFinalAndSidecar:
    @pytest.mark.asyncio
    async def test_returns_when_results_dir_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.environment import OperatorEnvironment

        missing = tmp_path / "definitely-missing"
        monkeypatch.setattr(OperatorEnvironment.RESULTS, "DIR", missing)
        client = MagicMock()
        client.download_all_results = AsyncMock()
        state: dict[str, Any] = {
            "downloaded": None,
            "checkpoints": None,
            "metrics": None,
        }
        await _download_final_and_sidecar(
            progress_client=client,
            controller_host="host",
            dest_dir=tmp_path,
            state=state,
        )
        client.download_all_results.assert_not_called()

    @pytest.mark.asyncio
    async def test_primary_download_populates_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.environment import OperatorEnvironment

        monkeypatch.setattr(OperatorEnvironment.RESULTS, "DIR", tmp_path)
        # Force key file absence so sidecar is consulted, but pin sidecar port
        # equal to API_SERVICE so the sidecar branch is short-circuited.
        monkeypatch.setattr(
            K8sEnvironment.PORTS,
            "RESULTS_SIDECAR",
            K8sEnvironment.PORTS.API_SERVICE,
        )

        client = MagicMock()
        client.download_all_results = AsyncMock(
            return_value=[
                f"{CHECKPOINTS_DIR_NAME}/cp1.json",
                "extra.csv",
            ]
        )
        state: dict[str, Any] = {
            "downloaded": None,
            "checkpoints": None,
            "metrics": None,
        }

        await _download_final_and_sidecar(
            progress_client=client,
            controller_host="host",
            dest_dir=tmp_path,
            state=state,
        )

        assert state["downloaded"] == ["extra.csv"]
        assert state["checkpoints"] == [f"{CHECKPOINTS_DIR_NAME}/cp1.json"]

    @pytest.mark.asyncio
    async def test_skips_sidecar_when_key_file_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.environment import OperatorEnvironment

        monkeypatch.setattr(OperatorEnvironment.RESULTS, "DIR", tmp_path)
        # Distinct ports so the would-be-sidecar branch is the only thing
        # gating the second client open.
        monkeypatch.setattr(K8sEnvironment.PORTS, "RESULTS_SIDECAR", 9999)
        monkeypatch.setattr(K8sEnvironment.PORTS, "API_SERVICE", 1234)

        # Sidecar client must NOT be opened. Patch ProgressClient on the
        # _completion_fetch module so we can detect any sidecar instantiation.
        sidecar_factory = MagicMock()
        monkeypatch.setattr(cf, "ProgressClient", sidecar_factory)

        client = MagicMock()
        client.download_all_results = AsyncMock(
            return_value=["profile_export_aiperf.json"]
        )
        state: dict[str, Any] = {
            "downloaded": None,
            "checkpoints": None,
            "metrics": None,
        }

        await _download_final_and_sidecar(
            progress_client=client,
            controller_host="host",
            dest_dir=tmp_path,
            state=state,
        )

        assert state["downloaded"] == ["profile_export_aiperf.json"]
        sidecar_factory.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_sidecar_when_port_collides_with_api_service(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.environment import OperatorEnvironment

        monkeypatch.setattr(OperatorEnvironment.RESULTS, "DIR", tmp_path)
        # Same port -> consolidated, no sidecar.
        monkeypatch.setattr(K8sEnvironment.PORTS, "RESULTS_SIDECAR", 1234)
        monkeypatch.setattr(K8sEnvironment.PORTS, "API_SERVICE", 1234)

        sidecar_factory = MagicMock()
        monkeypatch.setattr(cf, "ProgressClient", sidecar_factory)

        client = MagicMock()
        # No key file; would normally open sidecar — but port collision
        # short-circuits.
        client.download_all_results = AsyncMock(return_value=["nope.csv"])
        state: dict[str, Any] = {
            "downloaded": None,
            "checkpoints": None,
            "metrics": None,
        }

        await _download_final_and_sidecar(
            progress_client=client,
            controller_host="host",
            dest_dir=tmp_path,
            state=state,
        )

        sidecar_factory.assert_not_called()

    @pytest.mark.asyncio
    async def test_opens_sidecar_when_no_key_file_and_distinct_port(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.environment import OperatorEnvironment

        monkeypatch.setattr(OperatorEnvironment.RESULTS, "DIR", tmp_path)
        monkeypatch.setattr(K8sEnvironment.PORTS, "RESULTS_SIDECAR", 7777)
        monkeypatch.setattr(K8sEnvironment.PORTS, "API_SERVICE", 8888)

        # Stub sidecar context manager.
        sidecar_instance = MagicMock()
        sidecar_instance.download_all_results = AsyncMock(
            return_value=[
                "profile_export_aiperf.json",
                f"{CHECKPOINTS_DIR_NAME}/late.json",
            ]
        )
        sidecar_cm = MagicMock()
        sidecar_cm.__aenter__ = AsyncMock(return_value=sidecar_instance)
        sidecar_cm.__aexit__ = AsyncMock(return_value=None)
        sidecar_factory = MagicMock(return_value=sidecar_cm)
        monkeypatch.setattr(cf, "ProgressClient", sidecar_factory)

        # Primary returns a non-key file; sidecar provides the key file.
        client = MagicMock()
        client.download_all_results = AsyncMock(return_value=["misc.txt"])
        state: dict[str, Any] = {
            "downloaded": None,
            "checkpoints": None,
            "metrics": None,
        }

        await _download_final_and_sidecar(
            progress_client=client,
            controller_host="host",
            dest_dir=tmp_path,
            state=state,
        )

        # ProgressClient was instantiated with the sidecar port.
        sidecar_factory.assert_called_once_with(port=7777)
        assert state["downloaded"] == ["misc.txt", "profile_export_aiperf.json"]
        assert state["checkpoints"] == [f"{CHECKPOINTS_DIR_NAME}/late.json"]

    @pytest.mark.asyncio
    async def test_cancellation_between_primary_and_sidecar_preserves_primary_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.environment import OperatorEnvironment

        monkeypatch.setattr(OperatorEnvironment.RESULTS, "DIR", tmp_path)
        monkeypatch.setattr(K8sEnvironment.PORTS, "RESULTS_SIDECAR", 7777)
        monkeypatch.setattr(K8sEnvironment.PORTS, "API_SERVICE", 8888)
        cancellation_event = asyncio.Event()

        async def primary_download(_host: str, _dest: Path) -> list[str]:
            cancellation_event.set()
            return ["partial.json"]

        client = MagicMock()
        client.download_all_results = AsyncMock(side_effect=primary_download)
        sidecar_factory = MagicMock()
        monkeypatch.setattr(cf, "ProgressClient", sidecar_factory)
        state: dict[str, Any] = {
            "downloaded": None,
            "checkpoints": None,
            "metrics": {"kept": 1},
        }

        with pytest.raises(cf._FetchCancelled):
            await _download_final_and_sidecar(
                progress_client=client,
                controller_host="host",
                dest_dir=tmp_path,
                state=state,
                is_cancelled=cancellation_event.is_set,
                cancellation_event=cancellation_event,
            )

        assert state["downloaded"] == ["partial.json"]
        assert state["metrics"] == {"kept": 1}
        sidecar_factory.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancellation_interrupts_in_flight_primary_download(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.environment import OperatorEnvironment

        monkeypatch.setattr(OperatorEnvironment.RESULTS, "DIR", tmp_path)
        cancellation_event = asyncio.Event()
        download_started = asyncio.Event()
        download_cancelled = asyncio.Event()
        never = asyncio.Event()

        async def long_download(_host: str, _dest: Path) -> list[str]:
            download_started.set()
            try:
                await never.wait()
            finally:
                download_cancelled.set()
            return []

        client = MagicMock()
        client.download_all_results = AsyncMock(side_effect=long_download)
        state: dict[str, Any] = {
            "downloaded": ["earlier.json"],
            "checkpoints": None,
            "metrics": {"kept": 1},
        }
        task = asyncio.create_task(
            _download_final_and_sidecar(
                progress_client=client,
                controller_host="host",
                dest_dir=tmp_path,
                state=state,
                is_cancelled=cancellation_event.is_set,
                cancellation_event=cancellation_event,
            )
        )

        await download_started.wait()
        cancellation_event.set()
        with pytest.raises(cf._FetchCancelled):
            await task

        assert download_cancelled.is_set()
        assert state["downloaded"] == ["earlier.json"]
        assert state["metrics"] == {"kept": 1}


# =====================================================================
# _fetch_once_into_state
# =====================================================================


class TestFetchOnceIntoState:
    @pytest.mark.asyncio
    async def test_cancelled_at_start_returns_cancelled_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cf, "is_cancellation_requested", lambda _key: True)
        client = MagicMock()
        client.get_metrics = AsyncMock()
        client.download_all_results = AsyncMock()
        state: dict[str, Any] = {
            "metrics": None,
            "downloaded": None,
            "checkpoints": None,
        }

        out = await _fetch_once_into_state(
            key="ns/job",
            controller_host="h",
            dest_dir=tmp_path,
            progress_client=client,
            state=state,
        )
        assert out.error == "Cancelled by CR deletion"
        client.get_metrics.assert_not_called()
        client.download_all_results.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancellation_after_metrics_preserves_metrics_and_skips_download(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cancellation_event = asyncio.Event()
        monkeypatch.setattr(
            cf,
            "is_cancellation_requested",
            lambda _key: cancellation_event.is_set(),
        )

        async def get_metrics(_host: str) -> dict[str, int]:
            cancellation_event.set()
            return {"kept": 1}

        client = MagicMock()
        client.get_metrics = AsyncMock(side_effect=get_metrics)
        client.download_all_results = AsyncMock()
        state: dict[str, Any] = {
            "metrics": None,
            "downloaded": None,
            "checkpoints": None,
        }

        with pytest.raises(cf._FetchCancelled):
            await _fetch_once_into_state(
                key="ns/job",
                controller_host="h",
                dest_dir=tmp_path,
                progress_client=client,
                state=state,
                cancellation_event=cancellation_event,
            )

        assert state["metrics"] == {"kept": 1}
        client.download_all_results.assert_not_called()

    @pytest.mark.asyncio
    async def test_metrics_already_fetched_skips_get_metrics(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.environment import OperatorEnvironment

        monkeypatch.setattr(cf, "is_cancellation_requested", lambda _key: False)
        monkeypatch.setattr(OperatorEnvironment.RESULTS, "DIR", tmp_path)
        monkeypatch.setattr(
            K8sEnvironment.PORTS,
            "RESULTS_SIDECAR",
            K8sEnvironment.PORTS.API_SERVICE,
        )

        client = MagicMock()
        client.get_metrics = AsyncMock()
        client.download_all_results = AsyncMock(
            return_value=["profile_export_aiperf.json"]
        )
        state: dict[str, Any] = {
            "metrics": {"already": "have"},
            "downloaded": None,
            "checkpoints": None,
        }

        out = await _fetch_once_into_state(
            key="ns/job",
            controller_host="h",
            dest_dir=tmp_path,
            progress_client=client,
            state=state,
        )
        client.get_metrics.assert_not_called()
        assert out.metrics == {"already": "have"}
        assert out.downloaded == ["profile_export_aiperf.json"]

    @pytest.mark.asyncio
    async def test_downloads_without_key_file_raise_incomplete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.environment import OperatorEnvironment

        monkeypatch.setattr(cf, "is_cancellation_requested", lambda _key: False)
        monkeypatch.setattr(OperatorEnvironment.RESULTS, "DIR", tmp_path)
        monkeypatch.setattr(
            K8sEnvironment.PORTS,
            "RESULTS_SIDECAR",
            K8sEnvironment.PORTS.API_SERVICE,
        )

        client = MagicMock()
        client.get_metrics = AsyncMock(return_value={"k": "v"})
        client.download_all_results = AsyncMock(
            return_value=[f"{CHECKPOINTS_DIR_NAME}/c1.json", "misc.bin"]
        )
        state: dict[str, Any] = {
            "metrics": None,
            "downloaded": None,
            "checkpoints": None,
        }

        with pytest.raises(_IncompleteResultsError) as exc_info:
            await _fetch_once_into_state(
                key="ns/job",
                controller_host="h",
                dest_dir=tmp_path,
                progress_client=client,
                state=state,
            )
        # The raised exception carries the partial state.
        assert exc_info.value.metrics == {"k": "v"}
        assert exc_info.value.downloaded == ["misc.bin"]
        assert exc_info.value.checkpoints == [f"{CHECKPOINTS_DIR_NAME}/c1.json"]

    @pytest.mark.asyncio
    async def test_key_file_without_metrics_returns_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.environment import OperatorEnvironment

        monkeypatch.setattr(cf, "is_cancellation_requested", lambda _key: False)
        monkeypatch.setattr(OperatorEnvironment.RESULTS, "DIR", tmp_path)
        monkeypatch.setattr(
            K8sEnvironment.PORTS,
            "RESULTS_SIDECAR",
            K8sEnvironment.PORTS.API_SERVICE,
        )

        client = MagicMock()
        client.get_metrics = AsyncMock(return_value=None)
        client.download_all_results = AsyncMock(
            return_value=["profile_export_aiperf.json"]
        )
        state: dict[str, Any] = {
            "metrics": None,
            "downloaded": None,
            "checkpoints": None,
        }

        out = await _fetch_once_into_state(
            key="ns/job",
            controller_host="h",
            dest_dir=tmp_path,
            progress_client=client,
            state=state,
        )
        assert out.metrics is None
        assert out.downloaded == ["profile_export_aiperf.json"]
        assert out.error == ""

    @pytest.mark.asyncio
    async def test_metrics_and_key_file_returns_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.environment import OperatorEnvironment

        monkeypatch.setattr(cf, "is_cancellation_requested", lambda _key: False)
        monkeypatch.setattr(OperatorEnvironment.RESULTS, "DIR", tmp_path)
        monkeypatch.setattr(
            K8sEnvironment.PORTS,
            "RESULTS_SIDECAR",
            K8sEnvironment.PORTS.API_SERVICE,
        )

        client = MagicMock()
        client.get_metrics = AsyncMock(return_value={"k": "v"})
        client.download_all_results = AsyncMock(
            return_value=["profile_export_aiperf.json"]
        )
        state: dict[str, Any] = {
            "metrics": None,
            "downloaded": None,
            "checkpoints": None,
        }

        out = await _fetch_once_into_state(
            key="ns/job",
            controller_host="h",
            dest_dir=tmp_path,
            progress_client=client,
            state=state,
        )
        assert out.metrics == {"k": "v"}
        assert out.downloaded == ["profile_export_aiperf.json"]
        assert out.error == ""


# =====================================================================
# fetch_results_with_retry
# =====================================================================


class TestFetchResultsWithRetry:
    @pytest.mark.asyncio
    async def test_missing_dest_dir_and_body_raises_value_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            cf, "get_or_create_progress_client", AsyncMock(return_value=MagicMock())
        )
        with pytest.raises(ValueError, match="need either"):
            await fetch_results_with_retry(
                "host", "ns", "job", dest_dir=None, body=None
            )

    @pytest.mark.asyncio
    async def test_dest_dir_derived_from_body_when_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.environment import OperatorEnvironment

        monkeypatch.setattr(OperatorEnvironment.RESULTS, "DIR", tmp_path)
        monkeypatch.setattr(cf, "is_cancellation_requested", lambda _key: False)

        client = MagicMock()
        client.get_metrics = AsyncMock(return_value={"k": "v"})
        # Pre-populate the per-CR run dir so the download finds key file.
        client.download_all_results = AsyncMock(
            return_value=["profile_export_aiperf.json"]
        )
        monkeypatch.setattr(
            cf,
            "get_or_create_progress_client",
            AsyncMock(return_value=client),
        )
        monkeypatch.setattr(
            K8sEnvironment.PORTS,
            "RESULTS_SIDECAR",
            K8sEnvironment.PORTS.API_SERVICE,
        )

        # Need the run_dir to exist for _snapshot_bytes/results dir checks.
        body = {"metadata": {"creationTimestamp": "2024-04-25T18:22:03Z"}}
        from aiperf.operator.results_layout import epoch_key_from_body, run_dir

        expected_dir = run_dir(tmp_path, "ns", "job", epoch_key_from_body(body))
        expected_dir.mkdir(parents=True, exist_ok=True)

        out = await fetch_results_with_retry(
            "host", "ns", "job", dest_dir=None, body=body
        )
        assert out.error == ""
        assert out.metrics == {"k": "v"}
        # download_all_results was called with the derived dir.
        call_kwargs = client.download_all_results.call_args
        assert call_kwargs.args[1] == expected_dir

    @pytest.mark.asyncio
    async def test_explicit_dest_dir_used_as_is(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.environment import OperatorEnvironment

        monkeypatch.setattr(OperatorEnvironment.RESULTS, "DIR", tmp_path)
        monkeypatch.setattr(cf, "is_cancellation_requested", lambda _key: False)
        monkeypatch.setattr(
            K8sEnvironment.PORTS,
            "RESULTS_SIDECAR",
            K8sEnvironment.PORTS.API_SERVICE,
        )

        client = MagicMock()
        client.get_metrics = AsyncMock(return_value={"k": "v"})
        client.download_all_results = AsyncMock(
            return_value=["profile_export_aiperf.json"]
        )
        monkeypatch.setattr(
            cf, "get_or_create_progress_client", AsyncMock(return_value=client)
        )

        explicit = tmp_path / "explicit-target"
        explicit.mkdir()

        out = await fetch_results_with_retry(
            "host", "ns", "job", dest_dir=explicit, body=None
        )
        assert out.error == ""
        assert client.download_all_results.call_args.args[1] == explicit

    @pytest.mark.asyncio
    async def test_cancellation_at_top_returns_cancelled_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.environment import OperatorEnvironment

        monkeypatch.setattr(OperatorEnvironment.RESULTS, "DIR", tmp_path)
        # Cancellation is checked twice: once in _fetch_once_into_state, once
        # via the lambda in _run_fetch_loop_safely; both routes must return
        # a Cancelled-error result without touching get_metrics.
        monkeypatch.setattr(cf, "is_cancellation_requested", lambda _key: True)

        client = MagicMock()
        client.get_metrics = AsyncMock()
        client.download_all_results = AsyncMock()
        monkeypatch.setattr(
            cf, "get_or_create_progress_client", AsyncMock(return_value=client)
        )

        explicit = tmp_path / "target"
        explicit.mkdir()

        out = await fetch_results_with_retry(
            "host", "ns", "job", dest_dir=explicit, body=None
        )
        assert out.error == "Cancelled by CR deletion"
        client.get_metrics.assert_not_called()
        client.download_all_results.assert_not_called()
