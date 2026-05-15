# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for DAG-related fields on per-request records and BranchStats export."""

from __future__ import annotations

from unittest.mock import MagicMock

import orjson
import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.models import CreditPhaseStats
from aiperf.common.models.branch_stats import BranchStats
from aiperf.common.models.record_models import (
    MetricRecordMetadata,
    ProfileResults,
    RequestInfo,
    RequestRecord,
)
from aiperf.credit.messages import CreditPhaseCompleteMessage
from aiperf.records.record_processor_service import RecordProcessor
from aiperf.records.records_manager import RecordsManager


class TestMetricRecordMetadataDagFields:
    """Verify agent_depth / parent_correlation_id round-trip through serialization."""

    def test_defaults_when_not_provided(self):
        metadata = MetricRecordMetadata(
            session_num=1,
            request_start_ns=1,
            request_end_ns=2,
            worker_id="w",
            record_processor_id="p",
            benchmark_phase="profiling",
        )
        assert metadata.agent_depth == 0
        assert metadata.parent_correlation_id is None

    def test_roundtrip_with_dag_fields(self):
        metadata = MetricRecordMetadata(
            session_num=1,
            request_start_ns=1,
            request_end_ns=2,
            worker_id="w",
            record_processor_id="p",
            benchmark_phase="profiling",
            agent_depth=2,
            parent_correlation_id="parent-corr-id",
        )
        dumped = metadata.model_dump()
        assert dumped["agent_depth"] == 2
        assert dumped["parent_correlation_id"] == "parent-corr-id"

        restored = MetricRecordMetadata.model_validate(dumped)
        assert restored.agent_depth == 2
        assert restored.parent_correlation_id == "parent-corr-id"

    def test_json_serialization_includes_dag_fields(self):
        metadata = MetricRecordMetadata(
            session_num=1,
            request_start_ns=1,
            request_end_ns=2,
            worker_id="w",
            record_processor_id="p",
            benchmark_phase="profiling",
            agent_depth=3,
            parent_correlation_id="p",
        )
        as_json = orjson.loads(metadata.model_dump_json())
        assert as_json["agent_depth"] == 3
        assert as_json["parent_correlation_id"] == "p"


class TestRequestInfoDagFields:
    """Verify RequestInfo carries DAG fields for the worker -> record pipeline."""

    def test_request_info_defaults(self, sample_request_info: RequestInfo):
        assert sample_request_info.agent_depth == 0
        assert sample_request_info.parent_correlation_id is None

    def test_request_info_with_dag_fields(self, sample_request_info: RequestInfo):
        info = sample_request_info.model_copy(
            update={"agent_depth": 1, "parent_correlation_id": "parent-xyz"}
        )
        assert info.agent_depth == 1
        assert info.parent_correlation_id == "parent-xyz"

    def test_request_record_tagging_roundtrip(self, sample_request_info: RequestInfo):
        info = sample_request_info.model_copy(
            update={"agent_depth": 2, "parent_correlation_id": "p"}
        )
        record = RequestRecord(request_info=info)
        dumped = record.model_dump()
        assert dumped["request_info"]["agent_depth"] == 2
        assert dumped["request_info"]["parent_correlation_id"] == "p"


class TestMetricRecordMetadataFromRequestInfo:
    """_create_metric_record_metadata should propagate DAG fields from RequestInfo."""

    def test_propagates_dag_fields(self, sample_request_record: RequestRecord):
        sample_request_record.request_info = (
            sample_request_record.request_info.model_copy(
                update={
                    "agent_depth": 2,
                    "parent_correlation_id": "root-corr",
                }
            )
        )

        processor = MagicMock(spec=RecordProcessor)
        processor.service_id = "rp-1"

        metadata = RecordProcessor._create_metric_record_metadata(
            processor, sample_request_record, "worker-1"
        )

        assert metadata.agent_depth == 2
        assert metadata.parent_correlation_id == "root-corr"


class TestBranchStatsExport:
    """BranchStats serializes and lands in ProfileResults.branch_stats."""

    def test_branch_stats_defaults_all_zero(self):
        stats = BranchStats()
        dumped = stats.model_dump()
        assert dumped == {
            "children_spawned": 0,
            "children_completed": 0,
            "children_errored": 0,
            "parents_suspended": 0,
            "parents_resumed": 0,
            "parents_failed_due_to_child_error": 0,
            "joins_suppressed": 0,
            "children_truncated": 0,
        }

    def test_branch_stats_dict_helper(self):
        stats = BranchStats(
            children_spawned=5,
            children_completed=4,
            children_errored=1,
            parents_suspended=3,
            parents_resumed=3,
        )
        assert stats.stats_dict() == {
            "children_spawned": 5,
            "children_completed": 4,
            "children_errored": 1,
            "parents_suspended": 3,
            "parents_resumed": 3,
            "parents_failed_due_to_child_error": 0,
            "joins_suppressed": 0,
            "children_truncated": 0,
        }

    def test_branch_stats_roundtrip_through_profile_results(self):
        stats = BranchStats(
            children_spawned=2,
            children_completed=2,
            parents_suspended=1,
            parents_resumed=1,
        )
        results = ProfileResults(
            records=[],
            completed=0,
            start_ns=1,
            end_ns=2,
            branch_stats=stats,
        )

        as_json = orjson.loads(results.model_dump_json())
        assert as_json["branch_stats"]["children_spawned"] == 2
        assert as_json["branch_stats"]["parents_resumed"] == 1

        restored = ProfileResults.model_validate_json(results.model_dump_json())
        assert restored.branch_stats == stats

    def test_profile_results_omits_branch_stats_when_none(self):
        results = ProfileResults(records=[], completed=0, start_ns=1, end_ns=2)
        assert results.branch_stats is None
        # None-by-default survives a JSON roundtrip.
        restored = ProfileResults.model_validate_json(results.model_dump_json())
        assert restored.branch_stats is None


class TestRecordsManagerSnapshotBranchStats:
    """RecordsManager._snapshot_branch_stats returns stats stored per phase."""

    def test_snapshot_returns_none_when_phase_not_recorded(self):
        mgr = MagicMock(spec=RecordsManager)
        mgr._phase_branch_stats = {}
        assert RecordsManager._snapshot_branch_stats(mgr, CreditPhase.PROFILING) is None

    def test_snapshot_returns_stats_for_phase(self):
        stats = BranchStats(children_spawned=7, parents_resumed=2)
        mgr = MagicMock(spec=RecordsManager)
        mgr._phase_branch_stats = {CreditPhase.PROFILING: stats}

        snapshot = RecordsManager._snapshot_branch_stats(mgr, CreditPhase.PROFILING)
        assert snapshot is stats

    def test_snapshot_isolates_phases(self):
        warmup = BranchStats(children_spawned=1)
        profiling = BranchStats(children_spawned=5)
        mgr = MagicMock(spec=RecordsManager)
        mgr._phase_branch_stats = {
            CreditPhase.WARMUP: warmup,
            CreditPhase.PROFILING: profiling,
        }
        assert (
            RecordsManager._snapshot_branch_stats(mgr, CreditPhase.PROFILING)
            is profiling
        )
        assert RecordsManager._snapshot_branch_stats(mgr, CreditPhase.WARMUP) is warmup


class TestRecordsManagerOnCreditPhaseComplete:
    """Handler stores sub-agent stats from CreditPhaseCompleteMessage per phase."""

    @staticmethod
    def _make_phase_stats(
        phase: CreditPhase = CreditPhase.PROFILING,
    ) -> CreditPhaseStats:
        return CreditPhaseStats(
            phase=phase,
            requests_sent=10,
            requests_completed=10,
            final_requests_sent=10,
            start_ns=1_000_000,
        )

    @pytest.mark.asyncio
    async def test_stores_branch_stats_when_present(self):
        mgr = MagicMock(spec=RecordsManager)
        mgr._phase_branch_stats = {}
        mgr._complete_credit_phases = set()
        mgr._records_tracker = MagicMock()
        mgr._records_tracker.check_and_set_all_records_received_for_phase.return_value = False

        # Use WARMUP to skip the PROFILING-only logging branch that relies on
        # real phase_stats fields.
        phase_stats = self._make_phase_stats(CreditPhase.WARMUP)
        stats = BranchStats(children_spawned=4, parents_resumed=1)
        message = CreditPhaseCompleteMessage(
            service_id="tm-1",
            stats=phase_stats,
            branch_stats=stats,
        )

        await RecordsManager._on_credit_phase_complete(mgr, message)

        assert mgr._phase_branch_stats[phase_stats.phase] == stats

    @pytest.mark.asyncio
    async def test_no_op_when_branch_stats_absent(self):
        mgr = MagicMock(spec=RecordsManager)
        mgr._phase_branch_stats = {}
        mgr._complete_credit_phases = set()
        mgr._records_tracker = MagicMock()
        mgr._records_tracker.check_and_set_all_records_received_for_phase.return_value = False

        message = CreditPhaseCompleteMessage(
            service_id="tm-1",
            stats=self._make_phase_stats(CreditPhase.WARMUP),
        )

        await RecordsManager._on_credit_phase_complete(mgr, message)

        assert mgr._phase_branch_stats == {}
