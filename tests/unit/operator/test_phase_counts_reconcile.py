# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""status.phases must not contradict the exports on a successful run.

The phase counters mirror live controller progress, and at completion the
controller pod is racing its own shutdown. Three consecutive live gemma
sweeps put the same variation at 284/300, 300/300 and 240/300 for identical
work whose exports contained all 300 records every time. Once a run has
succeeded with real export files, those exports are authoritative.
"""

from __future__ import annotations

from pathlib import Path

import kopf
import orjson
import pytest
from pytest import param

from aiperf.kubernetes.crd_models import ControllerFetchResult
from aiperf.operator.handlers.completion import (
    _load_phase_manifest_payload,
    _reconcile_phase_counts_from_results,
    _ResultFlags,
)
from aiperf.operator.status import StatusBuilder


def _sb(profiling: dict | None) -> StatusBuilder:
    patch = kopf.Patch()
    if profiling is not None:
        patch.status["phases"] = {"profiling": profiling}
    return StatusBuilder(patch, {})


def _result(request_count: float | None) -> ControllerFetchResult:
    metrics = (
        {"request_count": {"avg": request_count, "unit": "requests"}}
        if request_count is not None
        else {}
    )
    return ControllerFetchResult(
        metrics=metrics,
        downloaded=["profile_export_aiperf.json"],
        checkpoints=[],
        error="",
    )


_OK = _ResultFlags(has_metrics=True, has_files=True, has_error=False, success=True)


def _manifest(*phases: dict) -> dict:
    return {"schema_version": 1, "phases": list(phases)}


class TestReconcilePhaseCounts:
    def test_manifest_reconciles_each_named_profiling_phase(self) -> None:
        patch = kopf.Patch()
        patch.status["phases"] = {
            "cache_prime": {
                "phaseName": "cache_prime",
                "phaseKind": "warmup",
                "requestsCompleted": 10,
            },
            "baseline": {
                "phaseName": "baseline",
                "phaseKind": "profiling",
                "requestsSent": 90,
                "requestsCompleted": 90,
                "requestsErrors": 1,
                "requestsTotal": 100,
                "recordsSuccess": 89,
                "recordsError": 1,
            },
            "main": {
                "phaseName": "main",
                "phaseKind": "profiling",
                "requestsSent": 190,
                "requestsCompleted": 190,
                "requestsErrors": 10,
                "requestsTotal": 200,
                "recordsSuccess": 180,
                "recordsError": 10,
            },
        }
        sb = StatusBuilder(patch, {})
        manifest = _manifest(
            {
                "phase_name": "cache_prime",
                "phase_kind": "warmup",
                "successful_request_count": 10,
                "error_request_count": 0,
                "total_request_count": 10,
            },
            {
                "phase_name": "baseline",
                "phase_kind": "profiling",
                "successful_request_count": 97,
                "error_request_count": 3,
                "total_request_count": 100,
            },
            {
                "phase_name": "main",
                "phase_kind": "profiling",
                "successful_request_count": 185,
                "error_request_count": 15,
                "total_request_count": 200,
            },
        )

        _reconcile_phase_counts_from_results(
            sb=sb,
            status={},
            result=_result(282),
            flags=_OK,
            phase_manifest=manifest,
        )

        baseline = patch.status["phases"]["baseline"]
        assert baseline["requestsSent"] == 100
        assert baseline["requestsCompleted"] == 100
        assert baseline["requestsErrors"] == 3
        assert baseline["recordsSuccess"] == 97
        assert baseline["recordsError"] == 3
        assert baseline["isRequestsComplete"] is True
        assert baseline["isRecordsComplete"] is True
        main = patch.status["phases"]["main"]
        assert main["requestsCompleted"] == 200
        assert main["recordsSuccess"] == 185
        assert main["recordsError"] == 15
        assert patch.status["phases"]["cache_prime"]["requestsCompleted"] == 10

    def test_manifest_keeps_live_counters_above_exported_counts(self) -> None:
        patch = kopf.Patch()
        patch.status["phases"] = {
            "main": {
                "phaseKind": "profiling",
                "requestsSent": 95,
                "requestsCompleted": 95,
                "requestsErrors": 6,
                "requestsTotal": 100,
                "recordsSuccess": 90,
                "recordsError": 6,
            }
        }
        sb = StatusBuilder(patch, {})

        _reconcile_phase_counts_from_results(
            sb=sb,
            status={},
            result=_result(85),
            flags=_OK,
            phase_manifest=_manifest(
                {
                    "phase_name": "main",
                    "phase_kind": "profiling",
                    "successful_request_count": 85,
                    "error_request_count": 5,
                    "total_request_count": 90,
                }
            ),
        )

        phase = patch.status["phases"]["main"]
        assert phase["requestsSent"] == 95
        assert phase["requestsCompleted"] == 95
        assert phase["requestsErrors"] == 6
        assert phase["recordsSuccess"] == 90
        assert phase["recordsError"] == 6

    def test_manifest_reconciles_duration_phase_with_zero_request_target(self) -> None:
        patch = kopf.Patch()
        patch.status["phases"] = {
            "timed": {
                "phaseKind": "profiling",
                "requestsCompleted": 8,
                "requestsTotal": 0,
                "recordsSuccess": 8,
            }
        }
        sb = StatusBuilder(patch, {})

        _reconcile_phase_counts_from_results(
            sb=sb,
            status={},
            result=_result(12),
            flags=_OK,
            phase_manifest=_manifest(
                {
                    "phase_name": "timed",
                    "phase_kind": "profiling",
                    "successful_request_count": 12,
                    "error_request_count": 0,
                    "total_request_count": 12,
                }
            ),
        )

        assert patch.status["phases"]["timed"]["requestsCompleted"] == 12
        assert patch.status["phases"]["timed"]["recordsSuccess"] == 12

    def test_invalid_manifest_uses_single_phase_aggregate_fallback(self) -> None:
        sb = _sb(
            {"requestsCompleted": 240, "requestsTotal": 300, "recordsSuccess": 240}
        )

        _reconcile_phase_counts_from_results(
            sb=sb,
            status={},
            result=_result(300),
            flags=_OK,
            phase_manifest=_manifest(
                {
                    "phase_name": "profiling",
                    "phase_kind": "profiling",
                    "successful_request_count": 299,
                    "error_request_count": 1,
                    "total_request_count": 299,
                }
            ),
        )

        assert sb._patch.status["phases"]["profiling"]["requestsCompleted"] == 300

    def test_custom_named_profiling_kind_is_reconciled(self) -> None:
        patch = kopf.Patch()
        patch.status["phases"] = {
            "cache_prime": {
                "phaseKind": "warmup",
                "requestsCompleted": 10,
            },
            "steady_state_profile": {
                "phaseKind": "profiling",
                "requestsCompleted": 240,
                "requestsTotal": 300,
                "recordsSuccess": 240,
            },
        }
        sb = StatusBuilder(patch, {})

        _reconcile_phase_counts_from_results(
            sb=sb, status={}, result=_result(300), flags=_OK
        )

        phase = patch.status["phases"]["steady_state_profile"]
        assert phase["requestsCompleted"] == 300
        assert phase["recordsSuccess"] == 300
        assert patch.status["phases"]["cache_prime"]["requestsCompleted"] == 10

    def test_multiple_profiling_phases_are_not_repaired_from_aggregate_count(
        self,
    ) -> None:
        patch = kopf.Patch()
        patch.status["phases"] = {
            "baseline": {
                "phaseKind": "profiling",
                "requestsCompleted": 90,
                "requestsTotal": 100,
            },
            "main": {
                "phaseKind": "profiling",
                "requestsCompleted": 190,
                "requestsTotal": 200,
            },
        }
        sb = StatusBuilder(patch, {})

        _reconcile_phase_counts_from_results(
            sb=sb, status={}, result=_result(300), flags=_OK
        )

        assert patch.status["phases"]["baseline"]["requestsCompleted"] == 90
        assert patch.status["phases"]["main"]["requestsCompleted"] == 190

    def test_lagging_counters_are_taken_from_the_exports(self) -> None:
        sb = _sb(
            {"requestsCompleted": 240, "requestsTotal": 300, "recordsSuccess": 240}
        )
        _reconcile_phase_counts_from_results(
            sb=sb, status={}, result=_result(300), flags=_OK
        )

        p = sb._patch.status["phases"]["profiling"]
        assert p["requestsCompleted"] == 300
        assert p["recordsSuccess"] == 300
        assert p["isRequestsComplete"] is True
        assert p["isRecordsComplete"] is True

    @pytest.mark.parametrize(
        "flags, reason",
        [
            param(
                _ResultFlags(has_metrics=True, has_files=True, has_error=True, success=False),
                "failed run keeps the counters showing how far it got",
                id="failed-run",
            ),
            param(
                _ResultFlags(has_metrics=False, has_files=True, has_error=False, success=True),
                "no parsed metrics means no authoritative request_count",
                id="no-metrics",
            ),
        ],
    )  # fmt: skip
    def test_left_alone_without_authoritative_success(
        self, flags: _ResultFlags, reason: str
    ) -> None:
        sb = _sb({"requestsCompleted": 240, "requestsTotal": 300})
        _reconcile_phase_counts_from_results(
            sb=sb, status={}, result=_result(300), flags=flags
        )
        assert sb._patch.status["phases"]["profiling"]["requestsCompleted"] == 240, (
            reason
        )

    def test_reconciles_when_downloaded_is_empty_but_metrics_parsed(self) -> None:
        """Sweep children finish with a populated summary and no download list.

        The exports were promoted from the operator's own disk rather than
        transferred in this call, so has_files is False while the parsed
        request_count is perfectly authoritative.
        """
        sb = _sb({"requestsCompleted": 256, "requestsTotal": 300})
        flags = _ResultFlags(
            has_metrics=True, has_files=False, has_error=False, success=True
        )
        _reconcile_phase_counts_from_results(
            sb=sb, status={}, result=_result(300), flags=flags
        )
        assert sb._patch.status["phases"]["profiling"]["requestsCompleted"] == 300

    def test_exports_exceeding_the_phase_target_are_not_papered_over(self) -> None:
        """That is a real inconsistency, not sampling lag."""
        sb = _sb({"requestsCompleted": 240, "requestsTotal": 300})
        _reconcile_phase_counts_from_results(
            sb=sb, status={}, result=_result(500), flags=_OK
        )
        assert sb._patch.status["phases"]["profiling"]["requestsCompleted"] == 240

    def test_current_counts_are_kept_but_missing_flags_are_set(self) -> None:
        """Counts already match, so only the completeness flags are filled in."""
        sb = _sb(
            {"requestsCompleted": 300, "requestsTotal": 300, "recordsSuccess": 300}
        )
        _reconcile_phase_counts_from_results(
            sb=sb, status={}, result=_result(300), flags=_OK
        )

        p = sb._patch.status["phases"]["profiling"]
        assert p["requestsCompleted"] == 300
        assert p["recordsSuccess"] == 300
        assert p["isRequestsComplete"] is True
        assert p["isRecordsComplete"] is True

    @pytest.mark.parametrize(
        "phases_value",
        [param(None, id="no-phases"), param({}, id="empty-phases")],
    )  # fmt: skip
    def test_missing_phase_block_is_a_no_op(self, phases_value: dict | None) -> None:
        sb = _sb(phases_value)
        _reconcile_phase_counts_from_results(
            sb=sb, status={}, result=_result(300), flags=_OK
        )

    def test_reads_existing_status_when_the_patch_has_no_phases(self) -> None:
        """The case that actually happens on a completed run.

        The final progress re-sample only writes phases into the patch when
        the controller was still answering; by completion it often is not, so
        the stale block lives only in the CR's existing status.
        """
        patch = kopf.Patch()
        sb = StatusBuilder(patch, {})
        existing = {
            "phases": {"profiling": {"requestsCompleted": 284, "requestsTotal": 300}}
        }
        _reconcile_phase_counts_from_results(
            sb=sb, status=existing, result=_result(300), flags=_OK
        )
        assert patch.status["phases"]["profiling"]["requestsCompleted"] == 300
        assert patch.status["phases"]["profiling"]["isRecordsComplete"] is True
        assert existing["phases"]["profiling"]["requestsCompleted"] == 284, (
            "the caller's status dict must not be mutated in place"
        )

    def test_missing_request_count_is_a_no_op(self) -> None:
        sb = _sb({"requestsCompleted": 240, "requestsTotal": 300})
        _reconcile_phase_counts_from_results(
            sb=sb, status={}, result=_result(None), flags=_OK
        )
        assert sb._patch.status["phases"]["profiling"]["requestsCompleted"] == 240

    def test_records_reconciled_even_when_requests_already_current(self) -> None:
        """The exact live failure: a tick landed between last request and last record.

        requestsCompleted was already 300 while recordsSuccess sat at 288, so
        an early return keyed on the request count skipped the records
        entirely and the CR kept isRecordsComplete=false forever.
        """
        sb = _sb(
            {
                "requestsCompleted": 300,
                "requestsTotal": 300,
                "isRequestsComplete": True,
                "recordsSuccess": 288,
                "isRecordsComplete": False,
            }
        )
        _reconcile_phase_counts_from_results(
            sb=sb, status={}, result=_result(300), flags=_OK
        )

        p = sb._patch.status["phases"]["profiling"]
        assert p["recordsSuccess"] == 300
        assert p["isRecordsComplete"] is True
        assert p["recordsProgressPercent"] == 100

    def test_fully_settled_phase_is_left_exactly_as_is(self) -> None:
        """Nothing to correct means nothing is altered."""
        settled = {
            "requestsSent": 300,
            "sendingComplete": True,
            "requestsCompleted": 300,
            "requestsTotal": 300,
            "isRequestsComplete": True,
            "recordsSuccess": 300,
            "isRecordsComplete": True,
        }
        sb = _sb(dict(settled))
        _reconcile_phase_counts_from_results(
            sb=sb, status={}, result=_result(300), flags=_OK
        )
        assert sb._patch.status["phases"]["profiling"] == settled

    def test_send_counters_are_reconciled_too(self) -> None:
        """Dispatch, completion and aggregation settle at three separate times.

        A tick landing between sent_end_ns and requests_end_ns leaves the CR
        claiming a finished run never finished sending.
        """
        sb = _sb(
            {
                "requestsSent": 284,
                "sendingComplete": False,
                "requestsCompleted": 300,
                "requestsTotal": 300,
                "isRequestsComplete": True,
                "recordsSuccess": 300,
                "isRecordsComplete": True,
            }
        )
        _reconcile_phase_counts_from_results(
            sb=sb, status={}, result=_result(300), flags=_OK
        )

        p = sb._patch.status["phases"]["profiling"]
        assert p["requestsSent"] == 300
        assert p["sendingComplete"] is True


def test_load_phase_manifest_payload_reads_controller_export(tmp_path: Path) -> None:
    payload = _manifest(
        {
            "phase_name": "main",
            "phase_kind": "profiling",
            "successful_request_count": 9,
            "error_request_count": 1,
            "total_request_count": 10,
        }
    )
    (tmp_path / "phase_manifest.json").write_bytes(orjson.dumps(payload))

    assert _load_phase_manifest_payload(tmp_path) == payload
