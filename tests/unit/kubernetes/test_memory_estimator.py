# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Kubernetes memory estimation framework."""

from __future__ import annotations

import pytest
from pytest import param

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
        requests = 1000
        est = _estimate_records_manager(requests, 25)

        # 24 scalar metrics + 3 timestamps + 4 numeric metadata columns,
        # plus 6 int32 categorical code columns and 2 uint8 booleans.
        bytes_per_row = (24 + 3 + 4) * 8 + 6 * 4 + 2
        column_bytes = 1024 * bytes_per_row * 1.05
        intern_bytes = requests * 136
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


class TestGpuTelemetryEstimate:
    def test_disabled(self) -> None:
        est = _estimate_gpu_telemetry(0, 300, 1.0, 12)
        assert est.variable_mib == 0
        assert "disabled" in est.formula

    def test_enabled(self) -> None:
        est = _estimate_gpu_telemetry(8, 300, 1.0, 12)
        assert est.variable_mib > 0
        assert est.name == "GPU Telemetry"

    def test_scales_with_gpus(self) -> None:
        few = _estimate_gpu_telemetry(2, 300, 1.0, 12)
        many = _estimate_gpu_telemetry(16, 300, 1.0, 12)
        assert many.variable_mib > few.variable_mib * 4

    def test_scales_with_duration(self) -> None:
        short = _estimate_gpu_telemetry(8, 60, 1.0, 12)
        long = _estimate_gpu_telemetry(8, 3600, 1.0, 12)
        assert long.variable_mib > short.variable_mib


class TestServerMetricsEstimate:
    def test_disabled(self) -> None:
        est = _estimate_server_metrics(
            0, 300, 5.0, unique_series=200, histogram_count=20, histogram_buckets=10
        )
        assert est.variable_mib == 0

    def test_enabled(self) -> None:
        est = _estimate_server_metrics(
            2, 300, 5.0, unique_series=200, histogram_count=20, histogram_buckets=10
        )
        assert est.variable_mib > 0

    def test_scales_with_endpoints(self) -> None:
        one = _estimate_server_metrics(
            1, 300, 5.0, unique_series=200, histogram_count=20, histogram_buckets=10
        )
        four = _estimate_server_metrics(
            4, 300, 5.0, unique_series=200, histogram_count=20, histogram_buckets=10
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
        params = _make_params(total_requests=1000)
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
        assert params.dataset_count == 500
        assert params.num_models == 1

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
        """Default connections_per_worker should be 200."""
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
                    {"name": "profiling", "type": "concurrency", "requests": 100}
                ],
            }
        )
        params = MemoryEstimationParams.from_config(config)
        assert params.connections_per_worker == 200


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
        """Streaming uses SSE chunks (200B/token), non-streaming uses text (4B/token)."""
        sse = _make_params(streaming=True, avg_osl_tokens=512)
        text = _make_params(streaming=False, avg_osl_tokens=512)
        est_sse = MemoryEstimator(sse).estimate()
        est_text = MemoryEstimator(text).estimate()
        wp_sse = est_sse.worker_pod.total_steady_state_mib
        wp_text = est_text.worker_pod.total_steady_state_mib
        # SSE should use noticeably more memory at OSL=512
        assert wp_sse > wp_text
