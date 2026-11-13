# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preflight for --goodput SLOs.

The record processors validate --goodput when they build their metric set, in a
service subprocess whose failure is not propagated: an SLO they reject left the
run sending every request and then waiting forever for records. The preflight
runs the same check in the main process, before any service spawns.
"""

from collections.abc import Generator
from types import SimpleNamespace

import pytest
from pytest import param

from aiperf.cli_runner._preflight import _preflight_goodput
from aiperf.config import BenchmarkConfig
from aiperf.config.loader.errors import ConfigurationError
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.metrics.types.good_request_count_metric import GoodRequestCountMetric
from tests.unit.conftest import make_benchmark_run


@pytest.fixture(autouse=True)
def _reset_slos() -> Generator[None, None, None]:
    yield
    GoodRequestCountMetric.set_slos({})


def _plan(*, streaming: bool, slos: dict[str, float] | None) -> SimpleNamespace:
    run = make_benchmark_run(
        endpoint_type="chat",
        streaming=streaming,
        extra={"slos": slos} if slos else None,
    )
    return SimpleNamespace(configs=[run.cfg])


class TestPreflightGoodput:
    @pytest.mark.parametrize(
        "streaming,slos",
        [
            param(False, {"time_to_first_token": 100.0}, id="ttft-without-streaming"),
            param(True, {"time_to_first_tokn": 100.0}, id="typo"),
            param(True, {"request_throughput": 1.0}, id="run-level-metric"),
            param(True, {"good_request_count": 1.0}, id="goodput-on-itself"),
        ],
    )  # fmt: skip
    def test_preflight_goodput_rejected_slo_raises_configuration_error(
        self, streaming: bool, slos: dict[str, float]
    ) -> None:
        with pytest.raises(ConfigurationError, match="Invalid --goodput"):
            _preflight_goodput(_plan(streaming=streaming, slos=slos))

    @pytest.mark.parametrize(
        "streaming,slos",
        [
            param(True, {"time_to_first_token": 100.0, "inter_token_latency": 5.0}, id="streaming-latency"),
            param(False, {"request_latency": 250.0}, id="non-streaming-latency"),
            param(False, None, id="no-slos"),
        ],
    )  # fmt: skip
    def test_preflight_goodput_applicable_slo_passes(
        self, streaming: bool, slos: dict[str, float] | None
    ) -> None:
        _preflight_goodput(_plan(streaming=streaming, slos=slos))

    def test_preflight_goodput_rejected_slo_leaves_no_slo_state(self) -> None:
        """A rejected SLO is not left installed on the shared metric class.

        An SLO on good_request_count makes it depend on itself, which would
        break later dependency ordering in the same process.
        """
        with pytest.raises(ConfigurationError, match="Invalid --goodput"):
            _preflight_goodput(_plan(streaming=True, slos={"good_request_count": 1.0}))
        assert GoodRequestCountMetric.required_metrics is None
        assert GoodRequestCountMetric._thresholds == {}
        MetricRegistry.create_dependency_order()

    def test_preflight_goodput_scenario_streaming_default_passes(self) -> None:
        """A scenario that turns streaming on when unset accepts a TTFT SLO."""
        cfg = BenchmarkConfig.model_validate(
            {
                "models": ["test-model"],
                "scenario": "inferencex-agentx-mvp",
                "endpoint": {"type": "chat", "urls": ["http://localhost:8000/v1"]},
                "datasets": [{"name": "default", "type": "synthetic"}],
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "concurrency": 1,
                        "requests": 1,
                    }
                ],
                "slos": {"time_to_first_token": 100.0},
            }
        )
        _preflight_goodput(SimpleNamespace(configs=[cfg]))
        assert cfg.endpoint.streaming is False  # the plan's config is not mutated
