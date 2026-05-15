# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""profile_export_aiperf.json includes BranchStats when DAG runs publish them.

DAG-shaped runs export BranchOrchestrator counters under
``branch_stats``; non-DAG runs leave the section out so existing
consumers don't see a spurious empty block.
"""

from pathlib import Path

import orjson
import pytest

from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.common.models import ProfileResults
from aiperf.common.models.branch_stats import BranchStats
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.exporters.metrics_json_exporter import MetricsJsonExporter
from aiperf.plugin.enums import EndpointType


@pytest.fixture
def user_config(tmp_path: Path) -> UserConfig:
    cfg = UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
            custom_endpoint="custom_endpoint",
        )
    )
    cfg.output.artifact_directory = tmp_path
    return cfg


def _make_exporter(
    user_config: UserConfig, branch_stats: BranchStats | None
) -> MetricsJsonExporter:
    results = ProfileResults(
        records=[],
        completed=0,
        start_ns=1_000_000_000,
        end_ns=2_000_000_000,
        branch_stats=branch_stats,
    )
    exporter_config = ExporterConfig(
        results=results,
        user_config=user_config,
        service_config=ServiceConfig(),
        telemetry_results=None,
    )
    return MetricsJsonExporter(exporter_config)


class TestProfileExportBranchStats:
    def test_branch_stats_in_json_when_present(self, user_config: UserConfig) -> None:
        stats = BranchStats(
            children_spawned=3,
            children_completed=3,
            children_errored=0,
            joins_suppressed=0,
        )
        exporter = _make_exporter(user_config, branch_stats=stats)
        content = exporter._generate_content()
        dumped = orjson.loads(content)

        assert "branch_stats" in dumped
        assert dumped["branch_stats"]["children_spawned"] == 3
        assert dumped["branch_stats"]["children_completed"] == 3
        assert dumped["branch_stats"]["joins_suppressed"] == 0

    def test_branch_stats_omitted_for_non_dag_runs(
        self, user_config: UserConfig
    ) -> None:
        exporter = _make_exporter(user_config, branch_stats=None)
        content = exporter._generate_content()
        dumped = orjson.loads(content)

        assert "branch_stats" not in dumped

    def test_branch_stats_round_trips_full_counters(
        self, user_config: UserConfig
    ) -> None:
        stats = BranchStats(
            children_spawned=10,
            children_completed=7,
            children_errored=2,
            children_truncated=1,
            parents_suspended=4,
            parents_resumed=4,
            parents_failed_due_to_child_error=0,
            joins_suppressed=2,
        )
        exporter = _make_exporter(user_config, branch_stats=stats)
        dumped = orjson.loads(exporter._generate_content())

        bs = dumped["branch_stats"]
        assert bs["children_spawned"] == 10
        assert bs["children_completed"] == 7
        assert bs["children_errored"] == 2
        assert bs["children_truncated"] == 1
        assert bs["parents_suspended"] == 4
        assert bs["parents_resumed"] == 4
        assert bs["parents_failed_due_to_child_error"] == 0
        assert bs["joins_suppressed"] == 2
