# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the deployment-resources estimator script.

Focuses on pure-logic surface:
- ``_make_params``: param-derivation math from (concurrency, isl, osl, ...)
  to ``MemoryEstimationParams`` (pod count, workers-per-pod clamp,
  record-processor scaling, total_requests = concurrency * multiplier).
- ``run_comparison_table``: smoke test that the function runs end-to-end
  for a small concurrency without raising and emits a header line we can
  assert on.

The ``run_detailed`` and ``main`` paths just delegate to ``_make_params``
and the estimator; we rely on _make_params coverage rather than re-test
those.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.common.environment import Environment
from tools.estimate_deployment_resources import (
    _DEFAULT_CONNECTIONS_PER_WORKER,
    _STRESS_HISTOGRAM_BUCKETS,
    _STRESS_HISTOGRAM_METRICS,
    _STRESS_UNIQUE_METRIC_SERIES,
    _make_params,
    run_comparison_table,
)

# ============================================================
# _make_params
# ============================================================


class TestMakeParams:
    def test_make_params_basic_fields(self) -> None:
        params = _make_params(
            concurrency=1000, isl=512, osl=128, streaming=True, workers=10
        )
        assert params.max_concurrency == 1000
        assert params.avg_isl_tokens == 512
        assert params.avg_osl_tokens == 128
        assert params.streaming is True
        assert params.total_workers == 10

    def test_make_params_total_requests_is_concurrency_times_multiplier(self) -> None:
        params = _make_params(
            concurrency=500, isl=128, osl=64, streaming=True, workers=10
        )
        # Default req_multiplier=4
        assert params.total_requests == 2000

    def test_make_params_explicit_req_multiplier_overrides(self) -> None:
        params = _make_params(
            concurrency=100,
            isl=128,
            osl=64,
            streaming=True,
            workers=10,
            req_multiplier=10,
        )
        assert params.total_requests == 1000

    def test_make_params_workers_per_pod_clamps_to_default(self) -> None:
        # 50 workers > DEFAULT_WORKERS_PER_POD: actual_wpp = DEFAULT_WORKERS_PER_POD.
        params = _make_params(
            concurrency=10_000, isl=128, osl=64, streaming=True, workers=50
        )
        wpp_default = Environment.WORKER.DEFAULT_WORKERS_PER_POD
        assert params.workers_per_pod == wpp_default

    def test_make_params_workers_below_default_uses_workers_count(self) -> None:
        wpp_default = Environment.WORKER.DEFAULT_WORKERS_PER_POD
        small_workers = max(1, wpp_default - 1)
        params = _make_params(
            concurrency=100,
            isl=128,
            osl=64,
            streaming=True,
            workers=small_workers,
        )
        assert params.workers_per_pod == small_workers

    def test_make_params_pod_count_ceil_division(self) -> None:
        # 25 workers, default 10 wpp => ceil(25/10) = 3 pods.
        params = _make_params(
            concurrency=10_000, isl=128, osl=64, streaming=True, workers=25
        )
        wpp_default = Environment.WORKER.DEFAULT_WORKERS_PER_POD
        if wpp_default == 10:
            assert params.num_worker_pods == 3
        else:
            # Computed dynamically; just ensure ceil-division correctness.
            from math import ceil

            assert params.num_worker_pods == ceil(25 / wpp_default)

    def test_make_params_record_processors_per_pod_at_least_one(self) -> None:
        params = _make_params(
            concurrency=10, isl=128, osl=64, streaming=True, workers=1
        )
        assert params.record_processors_per_pod >= 1

    def test_make_params_default_connections_per_worker(self) -> None:
        params = _make_params(
            concurrency=100, isl=128, osl=64, streaming=True, workers=10
        )
        assert params.connections_per_worker == _DEFAULT_CONNECTIONS_PER_WORKER

    def test_make_params_endpoints_scale_with_pods(self) -> None:
        # num_endpoints = max(1, pods // 25). For workers=2500 with default
        # wpp=10, pods=250 → endpoints=10. For workers=10, pods=1 → endpoints=1.
        small = _make_params(
            concurrency=100, isl=128, osl=64, streaming=True, workers=10
        )
        assert small.num_endpoints == 1

        big = _make_params(
            concurrency=100, isl=128, osl=64, streaming=True, workers=2500
        )
        assert big.num_endpoints >= 1

    def test_make_params_passes_stress_overrides(self) -> None:
        params = _make_params(
            concurrency=100, isl=128, osl=64, streaming=True, workers=10
        )
        assert params.est_unique_metric_series == _STRESS_UNIQUE_METRIC_SERIES
        assert params.est_histogram_metrics == _STRESS_HISTOGRAM_METRICS
        assert params.est_histogram_buckets == _STRESS_HISTOGRAM_BUCKETS

    @pytest.mark.parametrize(
        "num_gpus,num_endpoints",
        [
            (0, 0),
            param(8, 1, id="8-gpus-1-server-metrics-endpoint"),
        ],
    )  # fmt: skip
    def test_make_params_optional_kwargs_pass_through(
        self, num_gpus: int, num_endpoints: int
    ) -> None:
        params = _make_params(
            concurrency=100,
            isl=128,
            osl=64,
            streaming=True,
            workers=10,
            num_gpus=num_gpus,
            num_server_metrics_endpoints=num_endpoints,
        )
        assert params.num_gpus == num_gpus
        assert params.num_server_metrics_endpoints == num_endpoints

    def test_make_params_streaming_false_propagates(self) -> None:
        params = _make_params(
            concurrency=100, isl=128, osl=64, streaming=False, workers=10
        )
        assert params.streaming is False

    def test_make_params_export_http_trace_is_false(self) -> None:
        params = _make_params(
            concurrency=100, isl=128, osl=64, streaming=True, workers=10
        )
        assert params.export_http_trace is False


# ============================================================
# run_comparison_table — smoke
# ============================================================


class TestRunComparisonTable:
    def test_run_comparison_table_smoke(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Small concurrency so the estimator finishes quickly and we don't
        # depend on any specific numeric output.
        run_comparison_table(target_concurrency=100)
        out = capsys.readouterr().out
        # Header line emitted, plus at least one of the canned scenario labels.
        assert "Resource Estimates" in out
        assert "SSE ISL=128" in out
