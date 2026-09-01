# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Kubernetes memory estimation framework."""

from __future__ import annotations

import asyncio
import gc
import math
import tracemalloc
from collections.abc import Callable
from types import SimpleNamespace

import orjson
import pytest
from pytest import param

from aiperf.common.enums import CreditPhase
from aiperf.common.messages import MetricRecordsData
from aiperf.common.models import MetricRecordMetadata
from aiperf.common.models.dataset_models import Text, Turn
from aiperf.common.models.record_models import (
    RecordContext,
    RequestRecord,
    SSEField,
    SSEMessage,
    TextResponse,
)
from aiperf.kubernetes._memory_estimator.components import (
    _per_request_bytes,
    _rp_buffer_mib,
)
from aiperf.kubernetes._memory_estimator.constants import (
    _CATEGORICAL_INTERN_BYTES_PER_REQUEST,
    _COLUMN_STORE_INITIAL_CAPACITY,
    _COLUMN_STORE_LIST_METRIC_COLUMNS,
    _COLUMN_STORE_METADATA_BOOL_COLUMNS,
    _COLUMN_STORE_METADATA_CATEGORICAL_COLUMNS,
    _COLUMN_STORE_METADATA_NUMERIC_COLUMNS,
    _COLUMN_STORE_TIMESTAMP_COLUMNS,
    _FLOAT64_BYTES,
    _GROWABLE_ARRAY_OVERHEAD,
    _HEADROOM_WARNING_PCT,
    _INT32_BYTES,
    _SSE_BYTES_PER_CHUNK,
)
from aiperf.kubernetes.memory_estimator import (
    ClusterMemoryEstimate,
    ComponentEstimate,
    MemoryEstimationParams,
    MemoryEstimator,
    PodEstimate,
    _ceil_pow2,
    _estimate_dataset_manager,
    _estimate_fixed_service,
    _estimate_gpu_telemetry,
    _estimate_record_processor,
    _estimate_records_manager,
    _estimate_server_metrics,
    _estimate_worker,
    _mib,
    estimate_memory,
    format_estimate,
)
from aiperf.metrics.accumulator import MetricsAccumulator
from tests.unit.conftest import make_benchmark_run

# =============================================================================
# Utility tests
# =============================================================================


class TestCeilPow2:
    @pytest.mark.parametrize(
        "n, expected",
        [
            param(0, 1, id="zero"),
            param(1, 1, id="one"),
            param(2, 2, id="two"),
            param(3, 4, id="three"),
            param(4, 4, id="four"),
            param(5, 8, id="five"),
            param(255, 256, id="255"),
            param(256, 256, id="256"),
            param(257, 512, id="257"),
            param(1000, 1024, id="1000"),
            param(100_000, 131072, id="100k"),
            param(1_000_000, 1048576, id="1M"),
        ],
    )  # fmt: skip
    def test_ceil_pow2(self, n: int, expected: int) -> None:
        assert _ceil_pow2(n) == expected

    def test_ceil_pow2_negative(self) -> None:
        assert _ceil_pow2(-5) == 1


class TestMib:
    def test_bytes_to_mib(self) -> None:
        assert _mib(1024 * 1024) == 1.0

    def test_zero(self) -> None:
        assert _mib(0) == 0.0


# =============================================================================
# ComponentEstimate tests
# =============================================================================


class TestComponentEstimate:
    def test_steady_state(self) -> None:
        c = ComponentEstimate(
            name="test",
            base_mib=50,
            variable_mib=100,
            peak_mib=160,
            formula="",
            dominant_factor="",
        )
        assert c.steady_state_mib == 150

    def test_steady_state_zero_variable(self) -> None:
        c = ComponentEstimate(
            name="test",
            base_mib=50,
            variable_mib=0,
            peak_mib=50,
            formula="",
            dominant_factor="",
        )
        assert c.steady_state_mib == 50

    def test_headroom_pct_without_container_limit_is_infinite(self) -> None:
        c = ComponentEstimate(
            name="test",
            base_mib=0,
            variable_mib=0,
            peak_mib=500,
            formula="",
            dominant_factor="",
        )
        assert c.headroom_pct == math.inf
        assert c.at_risk is False

    def test_at_risk_true_when_peak_near_own_container_limit(self) -> None:
        c = ComponentEstimate(
            name="test",
            base_mib=0,
            variable_mib=0,
            peak_mib=175.0,
            formula="",
            dominant_factor="",
            container_limit_mib=192.0,
        )
        assert c.headroom_pct == pytest.approx(8.854, abs=1e-3)
        assert c.at_risk is True

    def test_at_risk_false_with_ample_container_headroom(self) -> None:
        c = ComponentEstimate(
            name="test",
            base_mib=0,
            variable_mib=0,
            peak_mib=100.0,
            formula="",
            dominant_factor="",
            container_limit_mib=192.0,
        )
        assert c.at_risk is False


# =============================================================================
# PodEstimate tests
# =============================================================================


class TestPodEstimate:
    def _make_pod(
        self, components: list[tuple[float, float, float]], limit: float
    ) -> PodEstimate:
        return PodEstimate(
            pod_type="test",
            components=[
                ComponentEstimate(
                    name=f"c{i}",
                    base_mib=base,
                    variable_mib=var,
                    peak_mib=peak,
                    formula="",
                    dominant_factor="",
                )
                for i, (base, var, peak) in enumerate(components)
            ],
            current_limit_mib=limit,
        )

    def test_total_steady_state(self) -> None:
        pod = self._make_pod([(50, 100, 160), (30, 20, 55)], 1024)
        assert pod.total_steady_state_mib == 200  # (50+100) + (30+20)

    def test_total_peak(self) -> None:
        pod = self._make_pod([(50, 100, 160), (30, 20, 55)], 1024)
        assert pod.total_peak_mib == 215  # 160 + 55

    def test_headroom_pct(self) -> None:
        pod = self._make_pod([(0, 0, 500)], 1000)
        assert pod.headroom_pct == pytest.approx(50.0)

    def test_headroom_zero_limit(self) -> None:
        pod = self._make_pod([(0, 0, 500)], 0)
        assert pod.headroom_pct == 0.0

    def test_at_risk_true(self) -> None:
        pod = self._make_pod([(0, 0, 900)], 1000)
        assert pod.at_risk is True  # 10% headroom < 15% threshold

    def test_at_risk_false(self) -> None:
        pod = self._make_pod([(0, 0, 500)], 1000)
        assert pod.at_risk is False

    def test_recommended_request(self) -> None:
        pod = self._make_pod([(100, 0, 100)], 1000)
        # steady_state=100, * 1.2 = 120
        assert pod.recommended_request_mib == 120

    def test_recommended_limit(self) -> None:
        pod = self._make_pod([(0, 0, 100)], 1000)
        # peak=100, * 1.3 = 130
        assert pod.recommended_limit_mib == 130

    def _make_controller_pod(self) -> PodEstimate:
        """Nine-container controller pod where only one container is thin.

        system_controller sits at 175 MiB against its own 192Mi limit (~8.9%
        headroom), while the pod-wide sum of all nine container limits leaves
        plenty of room -- the exact shape that a pod-aggregate-only check
        reports as healthy right up until the cgroup OOM-kills that container.
        """
        return PodEstimate(
            pod_type="controller",
            components=[
                ComponentEstimate(
                    name="System Controller",
                    base_mib=0,
                    variable_mib=0,
                    peak_mib=175.0,
                    formula="",
                    dominant_factor="",
                    container_limit_mib=192.0,
                ),
                ComponentEstimate(
                    name="Timing Manager",
                    base_mib=0,
                    variable_mib=0,
                    peak_mib=40.0,
                    formula="",
                    dominant_factor="",
                    container_limit_mib=192.0,
                ),
            ],
            current_limit_mib=1792.0,
        )

    def test_pod_aggregate_headroom_hides_a_thin_container(self) -> None:
        pod = self._make_controller_pod()
        # 1792 limit vs 215 MiB summed peak -> the pod-wide view looks fine.
        assert pod.headroom_pct > _HEADROOM_WARNING_PCT

    def test_at_risk_true_when_single_container_near_own_limit(self) -> None:
        pod = self._make_controller_pod()
        assert pod.at_risk is True

    def test_at_risk_components_lists_only_the_thin_container(self) -> None:
        pod = self._make_controller_pod()
        assert [c.name for c in pod.at_risk_components] == ["System Controller"]

    def test_at_risk_components_empty_when_every_container_has_headroom(self) -> None:
        pod = self._make_pod([(0, 0, 100)], 1000)
        assert pod.at_risk_components == []
        assert pod.at_risk is False


# =============================================================================
# Component estimator tests
# =============================================================================


class TestRecordsManagerEstimate:
    def test_small_run(self) -> None:
        est = _estimate_records_manager(total_requests=1000, num_metrics=25)
        assert est.name == "RecordsManager"
        assert est.base_mib > 0
        assert est.variable_mib > 0
        assert est.warning is None  # small run, no warning

    def test_large_run_warns(self) -> None:
        # At 1M requests the metric arrays alone are ~260 MiB — need more to trigger 500 MiB warning
        est = _estimate_records_manager(total_requests=5_000_000, num_metrics=25)
        assert est.warning is not None
        assert "5,000,000" in est.warning

    def test_scales_with_requests(self) -> None:
        small = _estimate_records_manager(1_000, 25)
        large = _estimate_records_manager(100_000, 25)
        assert large.variable_mib > small.variable_mib * 10

    def test_scales_with_metrics(self) -> None:
        few = _estimate_records_manager(10_000, 10)
        many = _estimate_records_manager(10_000, 50)
        assert many.variable_mib > few.variable_mib

    def test_models_current_column_store_fixed_layout(self) -> None:
        """The row width the estimate assumes matches the constants it is built from.

        Deliberately derives ``bytes_per_row`` from the layout constants rather
        than restating their literal values: hardcoding ``(24 + 3 + 4) * 8``
        here made this test a mirror of ``constants.py`` that passed for any
        value the constants held. The independent check that those constants
        describe the real ``ColumnStore`` lives in
        ``TestColumnStoreMetadataColumnDrift``.
        """
        requests = 1000
        num_metrics = 25
        est = _estimate_records_manager(requests, num_metrics)

        scalar_metrics = num_metrics - _COLUMN_STORE_LIST_METRIC_COLUMNS
        float64_columns = (
            scalar_metrics
            + _COLUMN_STORE_TIMESTAMP_COLUMNS
            + _COLUMN_STORE_METADATA_NUMERIC_COLUMNS
        )
        bytes_per_row = (
            float64_columns * _FLOAT64_BYTES
            + _COLUMN_STORE_METADATA_CATEGORICAL_COLUMNS * _INT32_BYTES
            + _COLUMN_STORE_METADATA_BOOL_COLUMNS
        )
        column_bytes = (
            _COLUMN_STORE_INITIAL_CAPACITY * bytes_per_row * _GROWABLE_ARRAY_OVERHEAD
        )
        intern_bytes = requests * _CATEGORICAL_INTERN_BYTES_PER_REQUEST
        expected_mib = _mib(column_bytes + intern_bytes) + 1.0

        assert est.variable_mib == pytest.approx(expected_mib)

    def test_models_conversation_id_intern_table_bounded_by_dataset(self) -> None:
        requests = 10_000
        no_dataset = _estimate_records_manager(requests, 25, dataset_count=0)
        reused_dataset = _estimate_records_manager(requests, 25, dataset_count=1000)
        unique_dataset = _estimate_records_manager(requests, 25, dataset_count=20_000)

        assert reused_dataset.variable_mib - no_dataset.variable_mib == pytest.approx(
            _mib(1000 * 136)
        )
        assert unique_dataset.variable_mib - no_dataset.variable_mib == pytest.approx(
            _mib(requests * 136)
        )

    def test_ragged_icl_models_request_times_osl_storage(self) -> None:
        requests = 500_000
        osl = 512
        buffered = _estimate_records_manager(requests, 25, avg_osl=osl, streaming=False)
        ragged = _estimate_records_manager(
            requests,
            25,
            avg_osl=osl,
            streaming=True,
            list_metric_backend="ragged",
        )

        values_capacity = _ceil_pow2(requests * (osl - 1))
        offsets_capacity = _ceil_pow2(requests)
        expected_ragged_mib = _mib(
            values_capacity * (8 + 4) * 1.05 + offsets_capacity * 8
        )
        assert ragged.variable_mib - buffered.variable_mib == pytest.approx(
            expected_ragged_mib
        )

    def test_ragged_icl_requires_streaming(self) -> None:
        low_osl = _estimate_records_manager(
            100_000,
            25,
            avg_osl=2,
            streaming=False,
            list_metric_backend="ragged",
        )
        high_osl = _estimate_records_manager(
            100_000,
            25,
            avg_osl=4096,
            streaming=False,
            list_metric_backend="ragged",
        )
        assert high_osl.variable_mib == low_osl.variable_mib

    def test_streaming_single_token_response_has_no_icl_storage(self) -> None:
        buffered = _estimate_records_manager(100_000, 25, avg_osl=1, streaming=False)
        ragged = _estimate_records_manager(
            100_000,
            25,
            avg_osl=1,
            streaming=True,
            list_metric_backend="ragged",
        )
        tdigest = _estimate_records_manager(
            100_000,
            25,
            avg_osl=1,
            streaming=True,
            list_metric_backend="tdigest",
        )
        assert ragged.variable_mib == buffered.variable_mib
        assert tdigest.variable_mib == buffered.variable_mib

    def test_tdigest_icl_memory_is_bounded(self) -> None:
        def tdigest_overhead(requests: int, osl: int) -> float:
            buffered = _estimate_records_manager(
                requests, 25, avg_osl=osl, streaming=False
            )
            streaming = _estimate_records_manager(
                requests,
                25,
                avg_osl=osl,
                streaming=True,
                list_metric_backend="tdigest",
            )
            return streaming.variable_mib - buffered.variable_mib

        assert tdigest_overhead(1000, 8) == pytest.approx(_mib(4 * 1024))
        assert tdigest_overhead(5_000_000, 4096) == pytest.approx(_mib(4 * 1024))


class TestDatasetManagerEstimate:
    def test_small_dataset(self) -> None:
        est = _estimate_dataset_manager(100, 512, 128, 1)
        assert est.name == "DatasetManager"
        assert est.base_mib > 0
        # Peak should be higher than steady state (generation spike)
        assert est.peak_mib > est.steady_state_mib

    def test_multi_turn_increases_peak(self) -> None:
        single = _estimate_dataset_manager(1000, 512, 128, 1)
        multi = _estimate_dataset_manager(1000, 512, 128, 5)
        assert multi.peak_mib > single.peak_mib

    def test_large_dataset_high_peak(self) -> None:
        est = _estimate_dataset_manager(100_000, 2048, 512, 3)
        # Should be significant
        assert est.peak_mib > 100


class TestWorkerEstimate:
    def test_basic(self) -> None:
        est = _estimate_worker(
            concurrency_per_worker=50,
            avg_osl=128,
            streaming=False,
            max_turns=1,
            avg_isl=512,
            connections_per_worker=500,
        )
        assert est.name == "Worker"
        assert est.base_mib > 0

    def test_streaming_more_memory(self) -> None:
        non_stream = _estimate_worker(
            50,
            128,
            streaming=False,
            max_turns=1,
            avg_isl=512,
            connections_per_worker=500,
        )
        stream = _estimate_worker(
            50,
            128,
            streaming=True,
            max_turns=1,
            avg_isl=512,
            connections_per_worker=500,
        )
        assert stream.variable_mib > non_stream.variable_mib

    def test_multi_turn_adds_sessions(self) -> None:
        single = _estimate_worker(
            50,
            128,
            streaming=False,
            max_turns=1,
            avg_isl=512,
            connections_per_worker=500,
        )
        multi = _estimate_worker(
            50,
            128,
            streaming=False,
            max_turns=5,
            avg_isl=512,
            connections_per_worker=500,
        )
        assert multi.variable_mib > single.variable_mib

    def test_high_concurrency(self) -> None:
        low = _estimate_worker(
            10,
            128,
            streaming=True,
            max_turns=1,
            avg_isl=512,
            connections_per_worker=500,
        )
        high = _estimate_worker(
            500,
            128,
            streaming=True,
            max_turns=1,
            avg_isl=512,
            connections_per_worker=500,
        )
        assert high.variable_mib > low.variable_mib


class TestRecordProcessorEstimate:
    def test_single_model(self) -> None:
        est = _estimate_record_processor(1)
        assert est.name == "RecordProcessor"
        assert est.variable_mib >= 150  # at least one tokenizer

    def test_multi_model_warns(self) -> None:
        est = _estimate_record_processor(4)
        assert est.warning is not None
        assert "4 models" in est.warning

    def test_scales_linearly(self) -> None:
        one = _estimate_record_processor(1)
        four = _estimate_record_processor(4)
        # Should be roughly 4x the tokenizer portion
        assert four.variable_mib > one.variable_mib * 3.5

    def test_streaming_raw_buffer_uses_shared_sse_chunk_cost(self) -> None:
        from aiperf.kubernetes._memory_estimator.constants import _SSE_BYTES_PER_CHUNK

        raw_buffer_mib, _ = _rp_buffer_mib(100, streaming=True)

        assert raw_buffer_mib == _mib(10 * (1500 + 100 * _SSE_BYTES_PER_CHUNK))


class TestGpuTelemetryEstimate:
    def test_disabled(self) -> None:
        est = _estimate_gpu_telemetry(8, 300, 1.0, 12, enabled=False)
        assert est.base_mib == 0
        assert est.variable_mib == 0
        assert "disabled" in est.formula

    def test_enabled_without_gpu_sources_keeps_manager_base(self) -> None:
        est = _estimate_gpu_telemetry(0, 300, 1.0, 12, enabled=True)
        assert est.base_mib > 0
        assert est.variable_mib == 0

    def test_enabled(self) -> None:
        est = _estimate_gpu_telemetry(8, 300, 1.0, 12, enabled=True)
        assert est.variable_mib > 0
        assert est.name == "GPU Telemetry"

    def test_scales_with_gpus(self) -> None:
        few = _estimate_gpu_telemetry(2, 300, 1.0, 12, enabled=True)
        many = _estimate_gpu_telemetry(16, 300, 1.0, 12, enabled=True)
        assert many.variable_mib > few.variable_mib * 4

    def test_scales_with_duration(self) -> None:
        short = _estimate_gpu_telemetry(8, 60, 1.0, 12, enabled=True)
        long = _estimate_gpu_telemetry(8, 3600, 1.0, 12, enabled=True)
        assert long.variable_mib > short.variable_mib


class TestServerMetricsEstimate:
    def test_disabled(self) -> None:
        est = _estimate_server_metrics(
            2,
            300,
            5.0,
            enabled=False,
            unique_series=200,
            histogram_count=20,
            histogram_buckets=10,
        )
        assert est.base_mib == 0
        assert est.variable_mib == 0

    def test_enabled_without_sources_keeps_manager_base(self) -> None:
        est = _estimate_server_metrics(
            0,
            300,
            5.0,
            enabled=True,
            unique_series=200,
            histogram_count=20,
            histogram_buckets=10,
        )
        assert est.base_mib > 0
        assert est.variable_mib == 0

    def test_enabled(self) -> None:
        est = _estimate_server_metrics(
            2,
            300,
            5.0,
            enabled=True,
            unique_series=200,
            histogram_count=20,
            histogram_buckets=10,
        )
        assert est.variable_mib > 0

    def test_scales_with_endpoints(self) -> None:
        one = _estimate_server_metrics(
            1,
            300,
            5.0,
            enabled=True,
            unique_series=200,
            histogram_count=20,
            histogram_buckets=10,
        )
        four = _estimate_server_metrics(
            4,
            300,
            5.0,
            enabled=True,
            unique_series=200,
            histogram_count=20,
            histogram_buckets=10,
        )
        assert four.variable_mib > one.variable_mib * 3


class TestFixedServiceEstimate:
    def test_known_service(self) -> None:
        est = _estimate_fixed_service("system_controller")
        assert est.base_mib > 0
        assert est.variable_mib == 0
        assert est.peak_mib == est.base_mib

    def test_custom_display_name(self) -> None:
        est = _estimate_fixed_service("api_service", "API Service")
        assert est.name == "API Service"

    def test_unknown_service_fallback(self) -> None:
        est = _estimate_fixed_service("unknown_service")
        assert est.base_mib > 0  # uses fallback


# =============================================================================
# MemoryEstimationParams tests
# =============================================================================


def _make_params(**overrides: object) -> MemoryEstimationParams:
    """Create MemoryEstimationParams with sensible defaults for testing."""
    defaults = dict(
        total_workers=10,
        workers_per_pod=10,
        num_worker_pods=1,
        record_processors_per_pod=2,
        max_concurrency=100,
        total_requests=10_000,
        total_benchmark_duration_s=300.0,
        dataset_count=1000,
        avg_isl_tokens=512,
        avg_osl_tokens=128,
        max_turns=1,
        streaming=True,
        list_metric_backend="ragged",
        num_endpoints=1,
        connections_per_worker=500,
        gpu_telemetry_enabled=True,
        server_metrics_enabled=True,
        server_metrics_discovery_enabled=False,
        num_gpus=0,
        gpu_sample_interval_s=1.0,
        num_gpu_metrics=12,
        num_server_metrics_endpoints=0,
        server_metrics_scrape_interval_s=5.0,
        est_unique_metric_series=200,
        est_histogram_metrics=20,
        est_histogram_buckets=10,
        num_models=1,
        num_standard_metrics=25,
        export_http_trace=False,
    )
    defaults.update(overrides)
    return MemoryEstimationParams(**defaults)


# =============================================================================
# Full estimator tests
# =============================================================================


class TestMemoryEstimator:
    def test_basic_estimate(self) -> None:
        params = _make_params()
        est = MemoryEstimator(params).estimate()
        assert isinstance(est, ClusterMemoryEstimate)
        assert est.controller.pod_type == "controller"
        assert est.worker_pod.pod_type == "worker"
        assert est.operator.pod_type == "operator"

    def test_cluster_total_positive(self) -> None:
        params = _make_params()
        est = MemoryEstimator(params).estimate()
        assert est.total_cluster_mib > 0

    def test_controller_has_all_components(self) -> None:
        params = _make_params()
        est = MemoryEstimator(params).estimate()
        names = {c.name for c in est.controller.components}
        assert "RecordsManager" in names
        assert "DatasetManager" in names
        assert "ZMQ Proxies" in names

    def test_worker_pod_has_scaled_components(self) -> None:
        params = _make_params(workers_per_pod=10, record_processors_per_pod=2)
        est = MemoryEstimator(params).estimate()
        names = {c.name for c in est.worker_pod.components}
        assert "Workers (x10)" in names
        assert "RecordProcessors (x2)" in names
        assert "WorkerGroupManager" in names

    def test_worker_pod_replicas(self) -> None:
        params = _make_params(num_worker_pods=5)
        est = MemoryEstimator(params).estimate()
        assert est.worker_pod.replicas == 5

    def test_non_divisible_concurrency_models_busiest_worker_and_processor(
        self,
    ) -> None:
        params = _make_params(
            total_workers=3,
            workers_per_pod=3,
            record_processors_per_pod=2,
            max_concurrency=10,
        )
        est = MemoryEstimator(params).estimate()
        workers = next(c for c in est.worker_pod.components if c.name == "Workers (x3)")
        processors = next(
            c for c in est.worker_pod.components if c.name == "RecordProcessors (x2)"
        )

        expected_worker = _estimate_worker(
            4,
            params.avg_osl_tokens,
            streaming=params.streaming,
            max_turns=params.max_turns,
            avg_isl=params.avg_isl_tokens,
            connections_per_worker=params.connections_per_worker,
        )
        expected_processor = _estimate_record_processor(
            params.num_models,
            avg_isl=params.avg_isl_tokens,
            avg_osl=params.avg_osl_tokens,
            streaming=params.streaming,
            concurrency_per_rp=5,
        )

        assert workers.variable_mib == pytest.approx(expected_worker.variable_mib * 3)
        assert processors.variable_mib == pytest.approx(
            expected_processor.variable_mib * 2
        )

    def test_server_metrics_discovery_warns_about_unmodeled_endpoints(self) -> None:
        est = MemoryEstimator(
            _make_params(server_metrics_discovery_enabled=True)
        ).estimate()

        assert any("discovery may add endpoints" in warning for warning in est.warnings)

    def test_multi_turn_warning_uses_busiest_worker(self) -> None:
        threshold = MemoryEstimator(
            _make_params(max_turns=2, max_concurrency=300, total_workers=3)
        ).estimate()
        busiest = MemoryEstimator(
            _make_params(max_turns=2, max_concurrency=301, total_workers=3)
        ).estimate()

        assert not any("Multi-turn" in warning for warning in threshold.warnings)
        assert any("101 concurrent sessions" in warning for warning in busiest.warnings)

    def test_high_request_count_warning(self) -> None:
        params = _make_params(total_requests=1_000_000)
        est = MemoryEstimator(params).estimate()
        assert any("1,000,000" in w for w in est.warnings)

    def test_at_risk_controller_recommendation(self) -> None:
        """Enough requests to potentially blow up the controller."""
        params = _make_params(total_requests=5_000_000)
        est = MemoryEstimator(params).estimate()
        # Should have warnings about records manager or controller headroom
        assert len(est.warnings) > 0

    def test_no_gpu_no_server_metrics(self) -> None:
        params = _make_params(num_gpus=0, num_server_metrics_endpoints=0)
        est = MemoryEstimator(params).estimate()
        gpu = next(c for c in est.controller.components if c.name == "GPU Telemetry")
        sm = next(c for c in est.controller.components if c.name == "Server Metrics")
        assert gpu.variable_mib == 0
        assert sm.variable_mib == 0

    def test_multi_turn_increases_worker_memory(self) -> None:
        single = MemoryEstimator(_make_params(max_turns=1)).estimate()
        multi = MemoryEstimator(_make_params(max_turns=5)).estimate()
        assert (
            multi.worker_pod.total_steady_state_mib
            > single.worker_pod.total_steady_state_mib
        )

    def test_streaming_vs_non_streaming(self) -> None:
        stream = MemoryEstimator(_make_params(streaming=True)).estimate()
        no_stream = MemoryEstimator(_make_params(streaming=False)).estimate()
        assert (
            stream.worker_pod.total_steady_state_mib
            > no_stream.worker_pod.total_steady_state_mib
        )

    def test_adequate_headroom_recommendation(self) -> None:
        params = _make_params(
            total_requests=1000,
            gpu_telemetry_enabled=False,
            server_metrics_enabled=False,
        )
        est = MemoryEstimator(params).estimate()
        assert any("adequate" in r.lower() for r in est.recommendations)

    def test_http_trace_warning(self) -> None:
        params = _make_params(export_http_trace=True, total_requests=50_000)
        est = MemoryEstimator(params).estimate()
        assert any("trace" in w.lower() for w in est.warnings)


# =============================================================================
# Format tests
# =============================================================================


class TestFormatEstimate:
    def test_produces_string(self) -> None:
        params = _make_params()
        est = MemoryEstimator(params).estimate()
        output = format_estimate(est)
        assert isinstance(output, str)
        assert "Memory Estimation" in output

    def test_contains_topology(self) -> None:
        params = _make_params(num_worker_pods=3, workers_per_pod=10)
        est = MemoryEstimator(params).estimate()
        output = format_estimate(est)
        assert "3 worker pod(s)" in output
        assert "10 workers/pod" in output

    def test_contains_cluster_total(self) -> None:
        params = _make_params()
        est = MemoryEstimator(params).estimate()
        output = format_estimate(est)
        assert "Cluster Total" in output
        assert "TOTAL" in output

    def test_contains_components(self) -> None:
        params = _make_params()
        est = MemoryEstimator(params).estimate()
        output = format_estimate(est)
        assert "RecordsManager" in output
        assert "DatasetManager" in output

    def test_warnings_displayed(self) -> None:
        params = _make_params(total_requests=1_000_000)
        est = MemoryEstimator(params).estimate()
        output = format_estimate(est)
        assert "[!]" in output or "Warnings:" in output


# =============================================================================
# Integration: from_config
# =============================================================================


class TestFromConfig:
    """Test MemoryEstimationParams.from_config with a real AIPerfConfig."""

    def test_basic_config(self) -> None:
        from aiperf.common.environment import Environment
        from aiperf.config.config import AIPerfConfig

        config = AIPerfConfig(
            benchmark={
                "models": "test-model",
                "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
                "datasets": [
                    {
                        "name": "main",
                        "type": "synthetic",
                        "entries": 500,
                        "prompts": {"isl": 256, "osl": 64},
                        "turns": {"mean": 10},
                    }
                ],
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "concurrency": 32,
                        "requests": 5000,
                    }
                ],
            }
        )
        params = MemoryEstimationParams.from_config(config, total_workers=10)
        assert params.max_concurrency == 32
        assert params.total_requests == 5000
        assert params.avg_isl_tokens == 256
        assert params.avg_osl_tokens == 64
        assert params.max_turns == 10
        assert params.dataset_count == 500
        assert params.num_models == 1
        assert params.gpu_telemetry_enabled
        assert params.server_metrics_enabled
        assert params.server_metrics_discovery_enabled
        assert params.gpu_sample_interval_s == Environment.GPU.COLLECTION_INTERVAL
        assert (
            params.server_metrics_scrape_interval_s
            == Environment.SERVER_METRICS.COLLECTION_INTERVAL
        )

    def test_server_metrics_endpoint_union_deduplicates_normalized_urls(self) -> None:
        from aiperf.config.config import AIPerfConfig

        config = AIPerfConfig(
            benchmark={
                "models": "test-model",
                "endpoint": {
                    "urls": [
                        "http://server-a:8000",
                        "http://server-b:8000/v1/chat/completions",
                    ]
                },
                "server_metrics": {
                    "urls": [
                        "http://server-a:8000/metrics",
                        "http://server-b:8000/v1/chat/completions/metrics",
                    ]
                },
                "datasets": [{"name": "main", "type": "synthetic", "entries": 100}],
                "phases": [
                    {"name": "profiling", "type": "concurrency", "requests": 100}
                ],
            }
        )

        params = MemoryEstimationParams.from_config(config)
        assert params.num_server_metrics_endpoints == 2

    def test_multi_phase_max_concurrency(self) -> None:
        from aiperf.config.config import AIPerfConfig

        config = AIPerfConfig(
            benchmark={
                "models": "test-model",
                "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
                "datasets": [
                    {
                        "name": "main",
                        "type": "synthetic",
                        "entries": 100,
                        "prompts": {"isl": 128},
                    }
                ],
                "phases": [
                    {
                        "name": "warmup",
                        "kind": "warmup",
                        "type": "concurrency",
                        "concurrency": 4,
                        "requests": 100,
                    },
                    {
                        "name": "profiling",
                        "kind": "profiling",
                        "type": "concurrency",
                        "concurrency": 64,
                        "requests": 10000,
                    },
                ],
            }
        )
        params = MemoryEstimationParams.from_config(config)
        assert params.max_concurrency == 64
        assert params.total_requests == 10100  # 100 + 10000

    def test_streaming_flag_and_list_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.common.environment import Environment
        from aiperf.config.config import AIPerfConfig

        monkeypatch.setattr(Environment.METRICS, "LIST_BACKEND", "tdigest")

        config = AIPerfConfig(
            benchmark={
                "models": "test-model",
                "endpoint": {
                    "urls": ["http://localhost:8000/v1/chat/completions"],
                    "streaming": True,
                },
                "datasets": [
                    {
                        "name": "main",
                        "type": "synthetic",
                        "entries": 100,
                        "prompts": {"isl": 128},
                    }
                ],
                "phases": [
                    {"name": "profiling", "type": "concurrency", "requests": 100}
                ],
            }
        )
        params = MemoryEstimationParams.from_config(config)
        assert params.streaming is True
        assert params.list_metric_backend == "tdigest"

    def test_estimate_memory_end_to_end(self) -> None:
        from aiperf.config.config import AIPerfConfig

        config = AIPerfConfig(
            benchmark={
                "models": "test-model",
                "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
                "datasets": [
                    {
                        "name": "main",
                        "type": "synthetic",
                        "entries": 1000,
                        "prompts": {"isl": 512, "osl": 128},
                    }
                ],
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "concurrency": 100,
                        "requests": 50000,
                    }
                ],
            }
        )
        est = estimate_memory(config, total_workers=20, workers_per_pod=10)
        assert est.params.num_worker_pods == 2
        assert est.params.workers_per_pod == 10
        assert est.total_cluster_mib > 0
        assert est.controller.total_peak_mib > 0
        assert est.worker_pod.replicas == 2

    def test_rate_based_phase(self) -> None:
        from aiperf.config.config import AIPerfConfig

        config = AIPerfConfig(
            benchmark={
                "models": "test-model",
                "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
                "datasets": [
                    {
                        "name": "main",
                        "type": "synthetic",
                        "entries": 100,
                        "prompts": {"isl": 128},
                    }
                ],
                "phases": [
                    {
                        "name": "profiling",
                        "type": "poisson",
                        "rate": 50,
                        "duration": 120,
                    }
                ],
            }
        )
        params = MemoryEstimationParams.from_config(config)
        # rate=50 * duration=120 = 6000 requests
        assert params.total_requests == 6000

    def test_default_connections_per_worker(self) -> None:
        """Default connections_per_worker tracks DeploymentConfig's default."""
        from aiperf.config.config import AIPerfConfig
        from aiperf.kubernetes.spec_converter import DEFAULT_CONNECTIONS_PER_WORKER

        config = AIPerfConfig(
            benchmark={
                "models": "test-model",
                "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
                "datasets": [
                    {
                        "name": "main",
                        "type": "synthetic",
                        "entries": 100,
                        "prompts": {"isl": 128},
                    }
                ],
                "phases": [
                    {"name": "profiling", "type": "concurrency", "requests": 100}
                ],
            }
        )
        params = MemoryEstimationParams.from_config(config)
        assert params.connections_per_worker == DEFAULT_CONNECTIONS_PER_WORKER


# =============================================================================
# Configuration defaults
# =============================================================================


class TestConfigurationDefaults:
    """Test that the calibrated defaults are consistent across all components."""

    def test_k8s_processor_scale_factor_is_1(self) -> None:
        from aiperf.kubernetes.environment import K8sEnvironment

        assert K8sEnvironment.RECORD_PROCESSOR_SCALE_FACTOR == 1

    def test_connections_per_worker_is_100(self) -> None:
        from aiperf.config.deployment import DeploymentConfig

        assert DeploymentConfig().connections_per_worker == 100

    def test_workers_per_pod_is_10(self) -> None:
        from aiperf.common.environment import Environment

        assert Environment.WORKER.DEFAULT_WORKERS_PER_POD == 10

    def test_rp_per_pod_with_10_workers(self) -> None:
        """10 workers / scale_factor 1 = 10 RPs per pod."""
        from aiperf.common.environment import Environment
        from aiperf.kubernetes.environment import K8sEnvironment

        wpp = Environment.WORKER.DEFAULT_WORKERS_PER_POD
        sf = K8sEnvironment.RECORD_PROCESSOR_SCALE_FACTOR
        assert max(1, wpp // sf) == 10

    def test_pod_concurrency_at_defaults(self) -> None:
        """10 workers x 100 conc/worker = 1000 per pod."""
        from aiperf.common.environment import Environment
        from aiperf.config.deployment import DeploymentConfig

        wpp = Environment.WORKER.DEFAULT_WORKERS_PER_POD
        cpw = DeploymentConfig().connections_per_worker
        assert wpp * cpw == 1000

    def test_controller_pod_guaranteed_qos(self) -> None:
        from aiperf.kubernetes.environment import K8sEnvironment

        pod = K8sEnvironment.SYSTEM_CONTROLLER
        resources = pod.to_k8s_resources()
        assert resources["requests"] == resources["limits"]

    def test_worker_pod_guaranteed_qos(self) -> None:
        from aiperf.kubernetes.environment import K8sEnvironment

        pod = K8sEnvironment.WORKER_POD
        resources = pod.to_k8s_resources()
        assert resources["requests"] == resources["limits"]

    def test_phase_estimation_constants_exist(self) -> None:
        """Hoisted phase-estimation magic numbers stay at the calibrated values.

        Lifting these to constants is documentation-only — the computed
        request count and duration math must not change.
        """
        from aiperf.kubernetes._memory_estimator.constants import (
            _DEFAULT_PHASE_REQUEST_COUNT,
            _PHASE_AVG_SEC_PER_REQUEST,
        )

        assert _DEFAULT_PHASE_REQUEST_COUNT == 1000
        assert _PHASE_AVG_SEC_PER_REQUEST == 2.0


# =============================================================================
# Scaling scenarios
# =============================================================================


class TestScalingScenarios:
    """Test that the estimator produces reasonable results at various scales."""

    def test_records_manager_warning_uses_its_container_limit(self) -> None:
        estimate = MemoryEstimator(_make_params()).estimate()

        assert any("RecordsManager uses" in warning for warning in estimate.warnings)

    def test_100k_concurrency_flags_controller_ragged_icl_risk(self) -> None:
        """Worker pods fit, but 400K streaming records need a larger controller."""
        params = _make_params(
            max_concurrency=100_000,
            total_workers=500,
            workers_per_pod=10,
            num_worker_pods=50,
            record_processors_per_pod=1,
            total_requests=400_000,
            connections_per_worker=200,
        )
        est = MemoryEstimator(params).estimate()
        assert est.worker_pod.replicas == 50
        assert est.controller.at_risk
        assert not est.worker_pod.at_risk

    def test_1m_concurrency_scales_linearly(self) -> None:
        """1M conc = 500 pods. Same per-pod, just more pods."""
        params_100k = _make_params(
            max_concurrency=100_000,
            total_workers=500,
            workers_per_pod=10,
            num_worker_pods=50,
            record_processors_per_pod=1,
            total_requests=400_000,
        )
        params_1m = _make_params(
            max_concurrency=1_000_000,
            total_workers=5000,
            workers_per_pod=10,
            num_worker_pods=500,
            record_processors_per_pod=1,
            total_requests=4_000_000,
        )
        est_100k = MemoryEstimator(params_100k).estimate()
        est_1m = MemoryEstimator(params_1m).estimate()

        # Per-pod memory should be the same
        assert (
            abs(
                est_100k.worker_pod.total_steady_state_mib
                - est_1m.worker_pod.total_steady_state_mib
            )
            < 1.0
        )  # within 1 MiB

        # Cluster total should scale ~10x (500 pods vs 50)
        ratio = est_1m.total_cluster_mib / est_100k.total_cluster_mib
        assert 9.0 < ratio < 11.0

    def test_high_isl_osl_increases_worker_memory(self) -> None:
        """ISL=4096 OSL=2048 should use more per-pod memory than ISL=512 OSL=128.

        With realistic per-process Python baselines (~150 MiB/process,
        calibrated against the 2026-04-30 ISL/OSL memory sweep),
        the per-process base dominates at low concurrency. The variable
        component still moves measurably with ISL/OSL — at default
        ``_make_params`` (max_concurrency=100, total_workers=10, so 10
        in-flight per worker) it adds ~2-5 MiB per worker. We assert any
        increase rather than a percentage threshold.
        """
        small = _make_params(avg_isl_tokens=512, avg_osl_tokens=128, streaming=True)
        large = _make_params(avg_isl_tokens=4096, avg_osl_tokens=2048, streaming=True)
        est_s = MemoryEstimator(small).estimate()
        est_l = MemoryEstimator(large).estimate()
        assert (
            est_l.worker_pod.total_steady_state_mib
            > est_s.worker_pod.total_steady_state_mib
        )

    def test_rp_token_pressure_at_high_isl_osl(self) -> None:
        """At ISL+OSL > 10K, RP queue depth scales with token pressure."""
        low_tokens = _make_params(avg_isl_tokens=512, avg_osl_tokens=128)
        high_tokens = _make_params(avg_isl_tokens=50000, avg_osl_tokens=50000)
        est_low = MemoryEstimator(low_tokens).estimate()
        est_high = MemoryEstimator(high_tokens).estimate()
        rp_low = next(
            c for c in est_low.worker_pod.components if "RecordProcessor" in c.name
        )
        rp_high = next(
            c for c in est_high.worker_pod.components if "RecordProcessor" in c.name
        )
        # Token pressure should inflate the RP estimate significantly
        assert rp_high.steady_state_mib > rp_low.steady_state_mib * 3

    def test_streaming_vs_nonstreaming_worker_difference(self) -> None:
        """Streaming retains one SSEMessage per chunk; buffered keeps one TextResponse."""
        sse = _make_params(streaming=True, avg_osl_tokens=512)
        text = _make_params(streaming=False, avg_osl_tokens=512)
        est_sse = MemoryEstimator(sse).estimate()
        est_text = MemoryEstimator(text).estimate()
        wp_sse = est_sse.worker_pod.total_steady_state_mib
        wp_text = est_text.worker_pod.total_steady_state_mib
        # SSE should use noticeably more memory at OSL=512
        assert wp_sse > wp_text


# =============================================================================
# Per-object byte constants vs measured heap
# =============================================================================

_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def _sse_chunk_json(request_idx: int, chunk_idx: int) -> str:
    """One OpenAI-compatible streaming chunk, shaped like the mock server's.

    Mirrors ``tests/aiperf_mock_server/utils.py::stream_chat_completion``. Both
    indices are interpolated so every chunk value is unique and CPython string
    interning cannot deduplicate them into a flattering measurement.
    """
    return orjson.dumps(
        {
            "id": f"chatcmpl-{request_idx:08d}",
            "object": "chat.completion.chunk",
            "created": 1712345678,
            "model": _MODEL,
            "choices": [
                {"index": 0, "delta": {"content": f" tok{chunk_idx}"}},
            ],
        }
    ).decode()


def _amortized_heap_bytes(factory: Callable[[int], object], n: int) -> float:
    """Amortized marginal heap bytes per object built by ``factory``.

    ``tracemalloc`` snapshot diff across ``n`` simultaneously-live instances,
    after a warmup pass. Amortizing is the whole point: the one-off cost of
    shared field-name strings, Pydantic validator/schema objects and interned
    literals is paid by the first instance and must not be charged to the
    marginal one. ``sys.getsizeof`` cannot be used — it sees only the
    top-level object, missing ``__dict__`` / ``__pydantic_extra__`` and every
    referenced string.
    """
    warmup = [factory(i) for i in range(16)]
    del warmup
    gc.collect()

    tracemalloc.start()
    gc.collect()
    before = tracemalloc.take_snapshot()
    live = [factory(i) for i in range(n)]
    gc.collect()
    after = tracemalloc.take_snapshot()
    total = sum(stat.size_diff for stat in after.compare_to(before, "filename"))
    tracemalloc.stop()

    del live
    gc.collect()
    # Discount the holder list's own pointer storage, which the estimator's
    # per-request model does not claim to cover.
    return (total - n * 8) / n


def _build_inflight_request(
    i: int, avg_isl: int, avg_osl: int, *, streaming: bool
) -> tuple[RequestRecord, Turn]:
    """One in-flight request as a worker holds it: record + context + turn + responses."""
    if streaming:
        responses: list[SSEMessage] | list[TextResponse] = [
            SSEMessage(
                perf_ns=i * 1_000_000 + k,
                packets=[SSEField(name="data", value=_sse_chunk_json(i, k))],
            )
            for k in range(avg_osl)
        ]
    else:
        body = orjson.dumps(
            {
                "id": f"cmpl-{i:08d}",
                "choices": [{"message": {"content": "y" * (avg_osl * 4)}}],
            }
        ).decode()
        responses = [
            TextResponse(perf_ns=i, content_type="application/json", text=body)
        ]
    record = RequestRecord(
        request_info=RecordContext(
            credit_num=i,
            credit_phase=CreditPhase.PROFILING,
            conversation_id=f"conv-{i:08d}",
            turn_index=0,
            x_request_id=f"6f1b0c9e-{i:08d}",
            x_correlation_id=f"9a2d4f77-{i:08d}",
        ),
        model_name=_MODEL,
        status=200,
        responses=responses,
    )
    # ~4 chars per token, the ratio _TURN_BYTES_PER_TOKEN encodes.
    prompt = f"{i:08d} " + "word " * (avg_isl * 4 // 5)
    turn = Turn(role="user", texts=[Text(name="text", contents=[prompt])])
    return record, turn


class TestPerRequestBytesAgainstMeasuredHeap:
    """``_per_request_bytes`` must stay on the conservative side of reality.

    This is the guard that was missing when ``_REQUEST_RECORD_BASE_BYTES`` and
    ``_TURN_BASE_BYTES`` survived a ``msgspec.Struct`` -> Pydantic migration at
    2.2-5.5x low, and when ``_SSE_BYTES_PER_CHUNK`` stayed fitted against a
    one-message-many-packets shape the transport never builds. Both defects
    were invisible to every other test in this file because they all compare
    the estimator against itself.

    The assertion is one-sided-with-a-ceiling on purpose: the estimator feeds a
    memory *recommendation*, so predicting slightly high is correct and
    predicting low is the bug. The upper bound only catches a runaway
    over-estimate that would inflate recommended limits.

    Measurement is heap-shape-dependent, not timing-dependent, so it is stable
    across machines for a given CPython build. It is sensitive to the CPython
    version's object layout: a future interpreter that changes Pydantic model
    or dataclass footprints should re-measure rather than widen the band.
    """

    # Ratios measured 2026-08-24 on CPython 3.12 sat in 1.01-1.07x.
    _MIN_RATIO = 1.0
    _MAX_RATIO = 1.6

    @pytest.mark.parametrize(
        "avg_isl, avg_osl, streaming, instances",
        [
            param(512, 128, True, 200, id="streaming_isl512_osl128"),
            param(512, 128, False, 200, id="buffered_isl512_osl128"),
            param(1024, 1024, True, 50, id="streaming_isl1024_osl1024"),
            param(1024, 1024, False, 200, id="buffered_isl1024_osl1024"),
        ],
    )  # fmt: skip
    def test_prediction_is_conservative_against_tracemalloc(
        self, avg_isl: int, avg_osl: int, streaming: bool, instances: int
    ) -> None:
        measured = _amortized_heap_bytes(
            lambda i: _build_inflight_request(i, avg_isl, avg_osl, streaming=streaming),
            instances,
        )
        predicted = _per_request_bytes(avg_isl, avg_osl, streaming=streaming)
        ratio = predicted / measured

        assert self._MIN_RATIO <= ratio <= self._MAX_RATIO, (
            f"_per_request_bytes(ISL={avg_isl}, OSL={avg_osl}, "
            f"streaming={streaming}) predicts {predicted:,} B but "
            f"{instances} live instances measure {measured:,.0f} B "
            f"({ratio:.2f}x). Re-derive the per-object constants in "
            f"_memory_estimator/constants.py from this measurement."
        )

    def test_streaming_chunk_constant_matches_one_message_per_chunk(self) -> None:
        """The OSL slope must track the shape the transport actually builds.

        ``aiohttp_client`` appends one ``SSEMessage`` per wire chunk to
        ``RequestRecord.responses``, so the marginal cost per output token is a
        whole message (message + packets list + SSEField + JSON string), not a
        bare ``SSEField``. Fitting against one message holding many packets
        yields ~112 B/chunk and is what made the streaming path under-predict.
        """
        low, high = 128, 1024
        at_low = _amortized_heap_bytes(
            lambda i: _build_inflight_request(i, 0, low, streaming=True), 200
        )
        at_high = _amortized_heap_bytes(
            lambda i: _build_inflight_request(i, 0, high, streaming=True), 50
        )
        measured_slope = (at_high - at_low) / (high - low)

        assert measured_slope <= _SSE_BYTES_PER_CHUNK, (
            f"_SSE_BYTES_PER_CHUNK={_SSE_BYTES_PER_CHUNK} under-predicts the "
            f"measured {measured_slope:.0f} B per streamed chunk."
        )
        assert measured_slope * 1.5 >= _SSE_BYTES_PER_CHUNK


# =============================================================================
# ColumnStore layout drift guard
# =============================================================================


class TestColumnStoreMetadataColumnDrift:
    """``_COLUMN_STORE_METADATA_NUMERIC_COLUMNS`` vs the real ``ColumnStore``.

    Drives records through the real ``MetricsAccumulator.process_record`` (and
    therefore ``ColumnStore.ingest_metadata``) and counts the columns that were
    actually *allocated*. ``ingest_metadata`` allocates lazily, skipping any
    field whose value is None, so the resident count is workload-dependent and
    the constant models the default streaming single-turn shape.

    The test this replaced restated ``(24 + 3 + 4) * 8`` with a matching prose
    comment, so it would have passed for any value the constant held.
    """

    @staticmethod
    def _record(
        session_num: int,
        *,
        request_ack_ns: int | None,
        cancellation_time_ns: int | None = None,
    ) -> MetricRecordsData:
        """A record shaped like the worker emits.

        ``credit_issued_ns`` is always set because ``Credit.issued_at_ns`` is a
        non-optional field, so the worker populates it on every request
        regardless of phase type. ``session_num`` and ``turn_index`` are
        likewise always present. That leaves ``request_ack_ns`` (streaming
        only) and ``cancellation_time_ns`` (cancelled requests only) as the two
        workload-dependent columns.
        """
        return MetricRecordsData(
            metadata=MetricRecordMetadata(
                session_num=session_num,
                request_start_ns=1_000_000 + session_num,
                request_end_ns=2_000_000 + session_num,
                credit_issued_ns=900_000 + session_num,
                request_ack_ns=request_ack_ns,
                cancellation_time_ns=cancellation_time_ns,
                conversation_id=f"conv-{session_num}",
                turn_index=0,
                worker_id="worker-1",
                record_processor_id="rp-1",
                benchmark_phase=CreditPhase.PROFILING,
                x_request_id=f"req-{session_num}",
                x_correlation_id=f"corr-{session_num}",
            ),
            metrics={"request_latency": 1_000_000.0},
            error=None,
        )

    @staticmethod
    async def _allocated_numeric_columns(
        records: list[MetricRecordsData], *, streaming: bool
    ) -> set[str]:
        accumulator = MetricsAccumulator(make_benchmark_run(streaming=streaming))
        for record in records:
            await accumulator.process_record(record)
        return set(accumulator.column_store._metadata_numeric)

    def test_default_streaming_workload_allocates_the_modeled_column_count(
        self,
    ) -> None:
        """Default streaming single-turn: the shape the constant is calibrated for."""
        records = [self._record(i, request_ack_ns=1_500_000 + i) for i in range(4)]
        allocated = asyncio.run(
            self._allocated_numeric_columns(records, streaming=True)
        )

        assert len(allocated) == _COLUMN_STORE_METADATA_NUMERIC_COLUMNS, (
            f"ColumnStore allocated {len(allocated)} numeric metadata columns "
            f"({sorted(allocated)}) for the default streaming workload, but "
            f"_COLUMN_STORE_METADATA_NUMERIC_COLUMNS is "
            f"{_COLUMN_STORE_METADATA_NUMERIC_COLUMNS}. Update the constant "
            f"and the RecordsManager row-width formula together."
        )

    def test_buffered_workload_allocates_one_fewer_column(self) -> None:
        """No ``request_ack_ns`` on the buffered path, so that column never allocates."""
        records = [self._record(i, request_ack_ns=None) for i in range(4)]
        allocated = asyncio.run(
            self._allocated_numeric_columns(records, streaming=False)
        )

        assert len(allocated) == _COLUMN_STORE_METADATA_NUMERIC_COLUMNS - 1
        assert "request_ack_ns" not in allocated

    def test_cancellations_allocate_one_extra_column(self) -> None:
        """A single cancelled request lifts the resident count above the default."""
        records = [
            self._record(0, request_ack_ns=1_500_000),
            self._record(1, request_ack_ns=1_500_001, cancellation_time_ns=1_900_000),
        ]
        allocated = asyncio.run(
            self._allocated_numeric_columns(records, streaming=True)
        )

        assert len(allocated) == _COLUMN_STORE_METADATA_NUMERIC_COLUMNS + 1
        assert "cancellation_time_ns" in allocated

    def test_lazy_allocation_skips_none_valued_fields(self) -> None:
        """The mechanism itself: ``ingest_metadata`` allocates only non-None fields.

        This is why the constant is 4 rather than the 5 numeric fields
        ``process_record`` offers.
        """
        records = [self._record(0, request_ack_ns=None)]
        allocated = asyncio.run(
            self._allocated_numeric_columns(records, streaming=True)
        )

        offered = {
            "session_num",
            "credit_issued_ns",
            "request_ack_ns",
            "cancellation_time_ns",
            "turn_index",
        }
        assert allocated < offered, (
            "ingest_metadata allocated every offered numeric field; lazy "
            "allocation is the premise _COLUMN_STORE_METADATA_NUMERIC_COLUMNS "
            "is calibrated against."
        )


# =============================================================================
# Topology derivation agrees with the deployed JobSet
# =============================================================================


def _topology_config(**runtime: object) -> object:
    """An AIPerfConfig whose only interesting content is ``benchmark.runtime``."""
    from aiperf.config.config import AIPerfConfig

    return AIPerfConfig(
        benchmark={
            "models": "test-model",
            "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
            "datasets": [{"name": "main", "type": "synthetic", "entries": 100}],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "concurrency": 32,
                    "requests": 1000,
                }
            ],
            "runtime": runtime,
        }
    )


def _deployed_record_processors_per_pod(config: object, num_pods: int) -> int:
    """Record-processor containers the JobSet manifest puts in each worker pod."""
    from aiperf.kubernetes.jobset import AIPerfJobSetSpec

    spec = AIPerfJobSetSpec(
        name="bench",
        namespace="aiperf",
        job_id="job",
        image="aiperf:test",
        worker_replicas=num_pods,
        workers_per_pod=config.benchmark.runtime.workers_per_pod,
        record_processors_per_pod=config.benchmark.runtime.record_processors_per_pod,
    )
    worker_job = next(
        job
        for job in spec.to_k8s_manifest()["spec"]["replicatedJobs"]
        if job["name"].startswith("worker")
    )
    containers = worker_job["template"]["spec"]["template"]["spec"]["containers"]
    return sum(1 for c in containers if c["name"].startswith("record-processor"))


class TestTopologyRecordProcessorDerivation:
    """``_derive_topology`` must honor ``runtime.record_processors_per_pod``.

    The deployment path honors it (``spec_converter.apply_worker_config`` ->
    ``AIPerfJobSetSpec._resolve_record_processors_per_pod``), so an estimate
    derived only from ``RECORD_PROCESSOR_SCALE_FACTOR`` described a topology
    that is not the one deployed.
    """

    @pytest.mark.parametrize(
        "workers_per_pod,configured_rp_per_pod,expected_rp_per_pod",
        [
            param(4, 1, 1, id="configured_below_scale_factor_derivation"),
            param(4, 2, 2, id="configured_between_floor_and_derivation"),
            param(4, 8, 8, id="configured_above_scale_factor_derivation"),
            param(4, None, 4, id="unset_falls_back_to_scale_factor"),
            param(1, None, 1, id="unset_single_worker_hits_floor"),
        ],
    )  # fmt: skip
    def test_rp_per_pod_honors_config(
        self,
        workers_per_pod: int,
        configured_rp_per_pod: int | None,
        expected_rp_per_pod: int,
    ) -> None:
        runtime: dict[str, object] = {"workers_per_pod": workers_per_pod}
        if configured_rp_per_pod is not None:
            runtime["record_processors_per_pod"] = configured_rp_per_pod
        params = MemoryEstimationParams.from_config(
            _topology_config(**runtime), total_workers=workers_per_pod
        )
        assert params.record_processors_per_pod == expected_rp_per_pod

    def test_unset_respects_floor_when_scale_factor_exceeds_workers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A scale factor above workers-per-pod must still leave one RP."""
        from aiperf.kubernetes.environment import K8sEnvironment

        monkeypatch.setattr(K8sEnvironment, "RECORD_PROCESSOR_SCALE_FACTOR", 8)
        params = MemoryEstimationParams.from_config(
            _topology_config(workers_per_pod=4), total_workers=4
        )
        assert params.record_processors_per_pod == 1

    def test_configured_rp_per_pod_lowers_the_worker_pod_estimate(self) -> None:
        """The count is load-bearing: fewer RPs must mean a smaller estimate."""
        derived = estimate_memory(_topology_config(workers_per_pod=4), total_workers=8)
        configured = estimate_memory(
            _topology_config(workers_per_pod=4, record_processors_per_pod=1),
            total_workers=8,
        )
        assert configured.worker_pod.total_peak_mib < derived.worker_pod.total_peak_mib

    @pytest.mark.parametrize(
        "total_workers,runtime",
        [
            param(8, {"workers_per_pod": 4}, id="defaults"),
            param(8, {"workers_per_pod": 4, "record_processors_per_pod": 1}, id="rp_per_pod_set"),
            param(8, {"workers_per_pod": 4, "record_processors_per_pod": 3}, id="rp_per_pod_above_workers"),
            param(8, {"workers_per_pod": 4, "record_processors": 4}, id="rp_total_set"),
            param(10, {"workers_per_pod": 8}, id="non_divisible_collapses_to_one_pod"),
            param(5, {}, id="no_runtime_overrides"),
        ],
    )  # fmt: skip
    def test_estimated_topology_matches_deployed_topology(
        self, total_workers: int, runtime: dict[str, object]
    ) -> None:
        """The real invariant: the estimate describes the pods that get created.

        The operator runs ``apply_worker_config`` before preflight, so the
        estimator sees the same normalized config the JobSet builder does.
        """
        from aiperf.kubernetes.spec_converter import apply_worker_config

        config = _topology_config(**runtime)
        try:
            num_pods = apply_worker_config(config, total_workers)
        except ValueError:
            with pytest.raises(ValueError):
                MemoryEstimationParams.from_config(config, total_workers=total_workers)
            return
        params = MemoryEstimationParams.from_config(config, total_workers=total_workers)

        assert params.num_worker_pods == num_pods
        assert params.workers_per_pod == config.benchmark.runtime.workers_per_pod
        assert params.record_processors_per_pod == _deployed_record_processors_per_pod(
            config, num_pods
        )


# Mechanical parity matrix: any divergence between the estimator's topology
# derivation and the deployment path's normalizer must fail a test, not ship.
_PARITY_TOTAL_WORKERS = (1, 3, 5, 8, 10, 16, 17, 32)
_PARITY_RUNTIMES: tuple[tuple[str, dict[str, object]], ...] = (
    ("no_overrides", {}),
    ("wpp_1", {"workers_per_pod": 1}),
    ("wpp_4", {"workers_per_pod": 4}),
    ("wpp_8", {"workers_per_pod": 8}),
    ("wpp_4_rp_per_pod_1", {"workers_per_pod": 4, "record_processors_per_pod": 1}),
    ("wpp_8_rp_per_pod_16", {"workers_per_pod": 8, "record_processors_per_pod": 16}),
)
_PARITY_CASES = [
    param(total, dict(runtime), id=f"{label}_total_{total}")
    for total in _PARITY_TOTAL_WORKERS
    for label, runtime in _PARITY_RUNTIMES
]


class TestTopologyMirrorsDeployment:
    """``_derive_topology`` must equal ``spec_converter.apply_worker_config``.

    The estimator banner runs on a *raw* config on both CLI paths: ``kube
    profile`` estimates before any normalization, and ``kube generate``
    normalizes only on the ``--no-operator`` branch. A second, hand-rolled copy
    of the topology rules therefore reported pod counts the JobSet never
    created (notably the single-pod collapse for non-divisible totals). These
    tests are mechanical: they compare against the real normalizer rather than
    against hardcoded numbers, so any future change to one side fails here.
    """

    @pytest.mark.parametrize(
        "total_workers,runtime",
        _PARITY_CASES,
    )  # fmt: skip
    def test_derive_topology_raw_config_matches_apply_worker_config(
        self, total_workers: int, runtime: dict[str, object]
    ) -> None:
        from aiperf.kubernetes.spec_converter import apply_worker_config

        deployed = _topology_config(**runtime)
        try:
            num_pods = apply_worker_config(deployed, total_workers)
        except ValueError:
            # A total that is neither <= workers_per_pod nor a multiple of it is
            # rejected by the deployment path; the estimator must reject it too
            # rather than reporting a topology the JobSet will never create.
            with pytest.raises(ValueError):
                MemoryEstimationParams.from_config(
                    _topology_config(**runtime), total_workers=total_workers
                )
            return
        deployed_runtime = deployed.benchmark.runtime

        estimated = MemoryEstimationParams.from_config(
            _topology_config(**runtime), total_workers=total_workers
        )

        assert (
            estimated.num_worker_pods,
            estimated.workers_per_pod,
            estimated.record_processors_per_pod,
        ) == (
            num_pods,
            deployed_runtime.workers_per_pod,
            deployed_runtime.record_processors_per_pod,
        )

    @pytest.mark.parametrize(
        "total_workers,workers_per_pod",
        [
            param(10, 8, id="ten_workers_eight_per_pod"),
            param(17, 4, id="seventeen_workers_four_per_pod"),
            param(3, 2, id="three_workers_two_per_pod"),
        ],
    )  # fmt: skip
    def test_derive_topology_non_divisible_total_is_rejected(
        self, total_workers: int, workers_per_pod: int
    ) -> None:
        """A JobSet cannot express a partial final pod; the estimate must agree.

        Silently folding onto one pod would have the banner describe a
        single-node run that the deployment path refuses to create.
        """
        with pytest.raises(ValueError, match="not a multiple of"):
            MemoryEstimationParams.from_config(
                _topology_config(workers_per_pod=workers_per_pod),
                total_workers=total_workers,
            )

    @pytest.mark.parametrize(
        "total_workers,workers_per_pod,record_processors,expected_pods,expected_rp",
        [
            param(8, 4, 4, 2, 2, id="total_splits_evenly_across_two_pods"),
            param(16, 4, 12, 4, 3, id="total_splits_evenly_across_four_pods"),
            param(8, 4, 2, 2, 1, id="total_below_pod_count_multiple"),
            param(8, 8, 5, 1, 5, id="total_equals_workers_per_pod_single_pod"),
        ],
    )  # fmt: skip
    def test_derive_topology_record_processors_total_splits_across_pods(
        self,
        total_workers: int,
        workers_per_pod: int,
        record_processors: int,
        expected_pods: int,
        expected_rp: int,
    ) -> None:
        """An authored ``runtime.record_processors`` is a cluster-wide total."""
        params = MemoryEstimationParams.from_config(
            _topology_config(
                workers_per_pod=workers_per_pod, record_processors=record_processors
            ),
            total_workers=total_workers,
        )
        assert params.num_worker_pods == expected_pods
        assert params.record_processors_per_pod == expected_rp

    def test_derive_topology_indivisible_record_processors_total_raises(self) -> None:
        """The deployment path rejects it, so the estimate must not paper over it."""
        with pytest.raises(ValueError, match="must be divisible"):
            MemoryEstimationParams.from_config(
                _topology_config(workers_per_pod=4, record_processors=3),
                total_workers=8,
            )

    def test_derive_topology_leaves_caller_config_unmodified(self) -> None:
        """The banner must not mutate the spec the CLI is about to submit."""
        config = _topology_config(workers_per_pod=8)
        before = config.benchmark.runtime.model_dump()

        MemoryEstimationParams.from_config(config, total_workers=16)

        assert config.benchmark.runtime.model_dump() == before

    def test_derive_topology_explicit_workers_per_pod_overrides_config(self) -> None:
        params = MemoryEstimationParams.from_config(
            _topology_config(workers_per_pod=8), total_workers=8, workers_per_pod=4
        )
        assert params.workers_per_pod == 4
        assert params.num_worker_pods == 2

    @pytest.mark.parametrize(
        "total_workers",
        [
            param(0, id="zero"),
            param(-4, id="negative"),
        ],
    )  # fmt: skip
    def test_derive_topology_non_positive_total_clamps_to_one_pod(
        self, total_workers: int
    ) -> None:
        """Unvalidated CR specs reach the estimator; they must not divide by zero."""
        params = MemoryEstimationParams.from_config(
            _topology_config(workers_per_pod=4), total_workers=total_workers
        )
        assert params.num_worker_pods == 1
        assert params.workers_per_pod == 1


class TestConnectionsPerWorkerDefault:
    """The estimator's connection-pool default must track the deployment default."""

    @pytest.mark.parametrize(
        "func",
        [
            param(estimate_memory, id="estimate_memory"),
            param(MemoryEstimationParams.from_config, id="from_config"),
        ],
    )  # fmt: skip
    def test_connections_per_worker_default_matches_deployment_config(
        self, func: Callable[..., object]
    ) -> None:
        import inspect

        from aiperf.config.deployment import DeploymentConfig
        from aiperf.kubernetes.spec_converter import DEFAULT_CONNECTIONS_PER_WORKER

        default = inspect.signature(func).parameters["connections_per_worker"].default
        assert default == DEFAULT_CONNECTIONS_PER_WORKER
        assert (
            default == DeploymentConfig.model_fields["connections_per_worker"].default
        )

    def test_from_config_default_connections_reach_params(self) -> None:
        from aiperf.kubernetes.spec_converter import DEFAULT_CONNECTIONS_PER_WORKER

        params = MemoryEstimationParams.from_config(_topology_config())
        assert params.connections_per_worker == DEFAULT_CONNECTIONS_PER_WORKER


class TestPhaseRequestRateTracksAvgSecPerRequest:
    """The concurrency-driven request estimate is the inverse of the duration one.

    ``_estimate_phase_duration`` spends ``_PHASE_AVG_SEC_PER_REQUEST`` seconds
    per request per concurrent slot, so ``_estimate_phase_requests`` must
    complete one request per slot every ``_PHASE_AVG_SEC_PER_REQUEST`` seconds.
    That reciprocal was hardcoded as ``0.5`` (and mislabeled "~10 req/sec per
    concurrent slot"), free to drift the moment the constant is recalibrated.
    """

    @staticmethod
    def _concurrency_phase(duration: float, concurrency: int) -> object:
        from aiperf.config.phases import ConcurrencyPhase
        from aiperf.plugin.enums import PhaseType

        return ConcurrencyPhase(
            name="profiling",
            type=PhaseType.CONCURRENCY,
            duration=duration,
            concurrency=concurrency,
        )

    @pytest.mark.parametrize(
        "duration,concurrency",
        [
            param(60.0, 1, id="one_slot"),
            param(60.0, 32, id="thirty_two_slots"),
            param(3600.0, 256, id="long_phase_many_slots"),
            param(1.0, 1, id="sub_average_duration"),
        ],
    )  # fmt: skip
    def test_estimate_phase_requests_concurrency_driven_matches_constant(
        self, duration: float, concurrency: int
    ) -> None:
        from aiperf.kubernetes._memory_estimator.constants import (
            _PHASE_AVG_SEC_PER_REQUEST,
        )
        from aiperf.kubernetes._memory_estimator.params import _estimate_phase_requests

        phase = self._concurrency_phase(duration, concurrency)
        assert _estimate_phase_requests(phase, concurrency) == int(
            duration * concurrency / _PHASE_AVG_SEC_PER_REQUEST
        )

    def test_estimate_phase_requests_rejects_infinite_duration(self) -> None:
        from aiperf.kubernetes._memory_estimator.params import _estimate_phase_requests

        phase = self._concurrency_phase(float("inf"), 1)

        with pytest.raises(ValueError, match="infinite phase duration"):
            _estimate_phase_requests(phase, 1)

    @pytest.mark.parametrize(
        "avg_sec_per_request,expected_requests",
        [
            param(2.0, 960, id="calibrated_default"),
            param(1.0, 1920, id="faster_requests_raise_the_estimate"),
            param(4.0, 480, id="slower_requests_lower_the_estimate"),
            param(0.5, 3840, id="sub_second_requests"),
        ],
    )  # fmt: skip
    def test_estimate_phase_requests_recalibrated_constant_moves_the_estimate(
        self,
        monkeypatch: pytest.MonkeyPatch,
        avg_sec_per_request: float,
        expected_requests: int,
    ) -> None:
        """A hardcoded reciprocal would ignore the monkeypatch and fail here."""
        from aiperf.kubernetes._memory_estimator import params as params_module

        monkeypatch.setattr(
            params_module, "_PHASE_AVG_SEC_PER_REQUEST", avg_sec_per_request
        )
        phase = self._concurrency_phase(60.0, 32)
        assert params_module._estimate_phase_requests(phase, 32) == expected_requests

    def test_derive_http_trace_uses_artifacts_trace_flag(self) -> None:
        from aiperf.kubernetes._memory_estimator.params import _derive_http_trace

        config = SimpleNamespace(artifacts=SimpleNamespace(trace=True))

        assert _derive_http_trace(config) is True

    def test_estimate_phase_duration_accounts_for_sessions(self) -> None:
        from aiperf.kubernetes._memory_estimator.constants import (
            _PHASE_AVG_SEC_PER_REQUEST,
        )
        from aiperf.kubernetes._memory_estimator.params import _estimate_phase_duration

        phase = SimpleNamespace(duration=None, requests=None, rate=None, sessions=10)

        assert (
            _estimate_phase_duration(phase, 5)
            == 10 * 3 * _PHASE_AVG_SEC_PER_REQUEST / 5
        )

    @pytest.mark.parametrize(
        "requests,concurrency",
        [
            param(1000, 1, id="serial"),
            param(1024, 32, id="evenly_divisible"),
            param(4096, 256, id="large_fan_out"),
        ],
    )  # fmt: skip
    def test_estimate_phase_requests_round_trips_estimate_phase_duration(
        self, requests: int, concurrency: int
    ) -> None:
        """Duration -> requests -> duration must be a fixed point."""
        from aiperf.config.phases import ConcurrencyPhase
        from aiperf.kubernetes._memory_estimator.params import (
            _estimate_phase_duration,
            _estimate_phase_requests,
        )
        from aiperf.plugin.enums import PhaseType

        counted = ConcurrencyPhase(
            name="profiling",
            type=PhaseType.CONCURRENCY,
            requests=requests,
            concurrency=concurrency,
        )
        duration = _estimate_phase_duration(counted, concurrency)

        timed = self._concurrency_phase(duration, concurrency)
        assert _estimate_phase_requests(timed, concurrency) == requests
