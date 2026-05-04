# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SteadyStateAnalyzer."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from aiperf.common.accumulator_protocols import SummaryContext
from aiperf.common.exceptions import PluginDisabled
from aiperf.config import AIPerfConfig
from aiperf.plugin.enums import AccumulatorType
from aiperf.post_processors.steady_state_analyzer import (
    SteadyStateAnalyzer,
    SteadyStateSummary,
)
from tests.unit.post_processors.conftest import (
    _make_run,
    create_accumulator_with_metrics,
    create_metric_records_message,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user_config(
    enabled: bool = True,
    min_window_pct: float = 10.0,
    start_pct: float | None = None,
    end_pct: float | None = None,
) -> AIPerfConfig:
    """Create an AIPerfConfig with the given steady-state settings.

    K8s wires steady_state under ``artifacts``; this helper writes through to
    ``artifacts.steady_state`` so the SteadyStateAnalyzer constructor sees the
    flag via ``run.cfg.output.steady_state`` (output is an alias for artifacts).
    """
    ss_kwargs: dict[str, object] = {
        "enabled": enabled,
        "min_window_pct": min_window_pct,
    }
    if start_pct is not None:
        ss_kwargs["start_pct"] = start_pct
    if end_pct is not None:
        ss_kwargs["end_pct"] = end_pct

    return AIPerfConfig(
        benchmark={
            "models": ["test-model"],
            "endpoint": {
                "urls": ["http://localhost:8000/v1/completions"],
                "type": "completions",
                "streaming": False,
            },
            "datasets": [
                {
                    "name": "default",
                    "type": "synthetic",
                    "entries": 1,
                    "prompts": {"isl": 8, "osl": 8},
                }
            ],
            "phases": [
                {
                    "name": "default",
                    "type": "concurrency",
                    "requests": 1,
                    "concurrency": 1,
                }
            ],
            "artifacts": {"steady_state": ss_kwargs},
        }
    )


def _make_record_metric():
    """Create a simple RECORD metric class for testing."""
    from aiperf.common.enums import MetricType

    class FakeLatency:
        tag = "request_latency"
        type = MetricType.RECORD
        header = "Request Latency"
        unit = "ms"

        def derive_value(self, results):
            raise NotImplementedError

    return FakeLatency


async def _build_accumulator_with_records(
    mock_metric_registry: Mock,
    user_config: AIPerfConfig,
    records: list[tuple[int, int, int, float]],
) -> object:
    """Build and populate a MetricsAccumulator.

    Args:
        records: list of (session_num, start_ns, end_ns, latency_value)
    """
    metric_cls = _make_record_metric()
    acc = create_accumulator_with_metrics(user_config, metric_cls)

    for session_num, start_ns, end_ns, latency in records:
        msg = create_metric_records_message(
            session_num=session_num,
            request_start_ns=start_ns,
            request_end_ns=end_ns,
            results=[{"request_latency": latency}],
        )
        await acc.process_record(msg.to_data())

    return acc


def _make_summary_ctx(acc: object) -> SummaryContext:
    """Build a SummaryContext with the given MetricsAccumulator."""
    return SummaryContext(
        accumulators={AccumulatorType.METRIC_RESULTS: acc},
        start_ns=0,
        end_ns=0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSteadyStateAnalyzerDisabled:
    def test_disabled_config_raises_plugin_disabled(self) -> None:
        """Raises PluginDisabled when steady-state is disabled."""
        config = _make_user_config(enabled=False)
        with pytest.raises(PluginDisabled, match="disabled"):
            SteadyStateAnalyzer(run=_make_run(config))


class TestSteadyStateAnalyzerNoRecords:
    @pytest.mark.asyncio
    async def test_no_accumulator_raises(self) -> None:
        """Raises PluginDisabled when MetricsAccumulator is not in context."""
        ctx = SummaryContext()
        config = _make_user_config()
        ss = SteadyStateAnalyzer(run=_make_run(config))
        with pytest.raises(PluginDisabled, match="MetricsAccumulator not available"):
            await ss.summarize(ctx)

    @pytest.mark.asyncio
    async def test_empty_accumulator_raises(
        self, mock_metric_registry: Mock, mock_user_config: AIPerfConfig
    ) -> None:
        """Raises PluginDisabled when accumulator has no records."""
        metric_cls = _make_record_metric()
        acc = create_accumulator_with_metrics(mock_user_config, metric_cls)
        ctx = _make_summary_ctx(acc)

        config = _make_user_config()
        ss = SteadyStateAnalyzer(run=_make_run(config))
        with pytest.raises(PluginDisabled, match="No records"):
            await ss.summarize(ctx)


class TestSteadyStateAnalyzerConstantConcurrency:
    @pytest.mark.asyncio
    async def test_constant_concurrency_covers_full_range(
        self, mock_metric_registry: Mock, mock_user_config: AIPerfConfig
    ) -> None:
        """With all requests overlapping, window should cover most of the range."""
        # 50 fully-overlapping requests: start=0, end=1_000_000_000
        records = [(i, 0, 1_000_000_000, float(i * 10)) for i in range(50)]
        acc = await _build_accumulator_with_records(
            mock_metric_registry, mock_user_config, records
        )
        ctx = _make_summary_ctx(acc)

        config = _make_user_config()
        ss = SteadyStateAnalyzer(run=_make_run(config))
        result = await ss.summarize(ctx)

        assert isinstance(result, SteadyStateSummary)
        # Detection method should be cusum-based (not old "concurrency_threshold")
        assert "cusum" in result.window_metadata.detection_method
        assert result.window_metadata.total_requests == 50
        # All requests should be in steady state (constant concurrency)
        assert result.window_metadata.steady_state_requests == 50
        # fraction_retained should be 1.0
        assert result.window_metadata.fraction_retained == pytest.approx(1.0)
        # Effective concurrency metric: constant 50 → all stats ≈ 50, std ≈ 0
        conc = result.effective_concurrency
        assert conc.avg == pytest.approx(50.0, rel=0.01)
        assert conc.min == pytest.approx(50.0)
        assert conc.max == pytest.approx(50.0)
        assert conc.std == pytest.approx(0.0, abs=0.1)


class TestSteadyStateAnalyzerRamp:
    @pytest.mark.asyncio
    async def test_ramp_up_steady_ramp_down(
        self, mock_metric_registry: Mock, mock_user_config: AIPerfConfig
    ) -> None:
        """Linear ramp → steady → drain. Verify window excludes ramp regions."""
        records = []
        session = 0

        # Ramp-up: staggered starts, all end at 900
        for i in range(10):
            start = int(i * 10)
            records.append((session, start, 900, 100.0))
            session += 1

        # Steady state: all start at 100, all end at 900
        for _i in range(40):
            records.append((session, 100, 900, 100.0))
            session += 1

        # Ramp-down: all start at 100, staggered ends
        for i in range(10):
            end = 900 + int(i * 10)
            records.append((session, 100, end, 100.0))
            session += 1

        acc = await _build_accumulator_with_records(
            mock_metric_registry, mock_user_config, records
        )
        ctx = _make_summary_ctx(acc)

        config = _make_user_config(min_window_pct=10.0)
        ss = SteadyStateAnalyzer(run=_make_run(config))
        result = await ss.summarize(ctx)

        assert result.window_metadata.total_requests == 60
        # Some requests should be excluded
        assert result.window_metadata.steady_state_requests <= 60
        # Should still have a meaningful window
        assert result.window_metadata.steady_state_duration_ns > 0
        # Detection method should be cusum-based
        assert "cusum" in result.window_metadata.detection_method


class TestSteadyStateAnalyzerUserOverride:
    @pytest.mark.asyncio
    async def test_user_override_exact_boundaries(
        self, mock_metric_registry: Mock, mock_user_config: AIPerfConfig
    ) -> None:
        """User override: start_pct=10, end_pct=90 → verify exact boundaries."""
        # 10 sequential requests spanning 0-1000
        records = [(i, i * 100, (i + 1) * 100, float(i)) for i in range(10)]
        acc = await _build_accumulator_with_records(
            mock_metric_registry, mock_user_config, records
        )
        ctx = _make_summary_ctx(acc)

        config = _make_user_config(start_pct=10.0, end_pct=90.0)
        ss = SteadyStateAnalyzer(run=_make_run(config))
        result = await ss.summarize(ctx)

        assert result.window_metadata.detection_method == "user_override"
        # Window should be 10%-90% of [0, 1000] = [100, 900]
        assert result.window_metadata.ramp_up_end_ns == pytest.approx(100.0)
        assert result.window_metadata.ramp_down_start_ns == pytest.approx(900.0)
        # Per-signal diagnostics should be None for user_override
        assert result.window_metadata.cusum_ramp_up_end_ns is None
        assert result.window_metadata.mser5_latency_ramp_up_end_ns is None
        # Effective concurrency metric should be populated
        conc = result.effective_concurrency
        assert conc.avg >= 0.0
        assert conc.min >= 0.0
        assert conc.max >= conc.min

    @pytest.mark.asyncio
    async def test_user_override_request_counts(
        self, mock_metric_registry: Mock, mock_user_config: AIPerfConfig
    ) -> None:
        """User override filters requests that fall entirely within the window."""
        # 10 sequential requests: [0,100), [100,200), ..., [900,1000)
        records = [(i, i * 100, (i + 1) * 100, float(i)) for i in range(10)]
        acc = await _build_accumulator_with_records(
            mock_metric_registry, mock_user_config, records
        )
        ctx = _make_summary_ctx(acc)

        # Window is [200, 800] — requests [2,3,4,5,6,7] have both endpoints inside
        config = _make_user_config(start_pct=20.0, end_pct=80.0)
        ss = SteadyStateAnalyzer(run=_make_run(config))
        result = await ss.summarize(ctx)

        assert result.window_metadata.steady_state_requests == 6


class TestSteadyStateAnalyzerMetricsCorrectness:
    @pytest.mark.asyncio
    async def test_windowed_metrics_match_subset(
        self, mock_metric_registry: Mock, mock_user_config: AIPerfConfig
    ) -> None:
        """Verify that metrics are computed only from steady-state records."""
        # 3 records: ramp(latency=1000), steady(latency=100), drain(latency=1000)
        records = [
            (0, 0, 100, 1000.0),  # ramp — latency 1000
            (1, 100, 900, 100.0),  # steady — latency 100
            (2, 900, 1000, 1000.0),  # drain — latency 1000
        ]
        acc = await _build_accumulator_with_records(
            mock_metric_registry, mock_user_config, records
        )
        ctx = _make_summary_ctx(acc)

        # Window is [10%, 90%] of [0, 1000] = [100, 900]
        config = _make_user_config(start_pct=10.0, end_pct=90.0)
        ss = SteadyStateAnalyzer(run=_make_run(config))
        result = await ss.summarize(ctx)

        # Only request 1 (latency=100) should be in the window
        assert result.window_metadata.steady_state_requests == 1
        assert "request_latency" in result.results
        assert result.results["request_latency"].avg == pytest.approx(100.0)


class TestSteadyStateSummarySerialize:
    def _make_summary(self) -> SteadyStateSummary:
        from aiperf.common.models import MetricResult
        from aiperf.post_processors.steady_state_analyzer import (
            SteadyStateWindowMetadata,
        )

        return SteadyStateSummary(
            results={
                "test_tag": MetricResult(
                    tag="test_tag", header="Test", unit="ms", avg=42.0, count=1
                )
            },
            effective_concurrency=MetricResult(
                tag="effective_concurrency",
                header="Effective Concurrency",
                unit="requests",
                avg=5.0,
                min=3.0,
                max=8.0,
                p50=5.0,
                p90=7.0,
                p95=8.0,
                p99=8.0,
                std=1.5,
            ),
            effective_throughput=MetricResult(
                tag="effective_throughput",
                header="Effective Throughput",
                unit="tokens/sec",
                avg=100.0,
                min=50.0,
                max=200.0,
                p50=100.0,
                p90=180.0,
                p95=190.0,
                p99=200.0,
                std=30.0,
            ),
            effective_prefill_throughput=MetricResult(
                tag="effective_prefill_throughput",
                header="Effective Prefill Throughput",
                unit="tokens/sec",
                avg=500.0,
                min=200.0,
                max=800.0,
                p50=500.0,
                p90=700.0,
                p95=750.0,
                p99=790.0,
                std=100.0,
            ),
            effective_generation_concurrency=MetricResult(
                tag="effective_generation_concurrency",
                header="Effective Generation Concurrency",
                unit="requests",
                avg=4.5,
                min=2.0,
                max=7.0,
                p50=4.5,
                p90=6.5,
                p95=7.0,
                p99=7.0,
                std=1.2,
            ),
            effective_prefill_concurrency=MetricResult(
                tag="effective_prefill_concurrency",
                header="Effective Prefill Concurrency",
                unit="requests",
                avg=3.0,
                min=1.0,
                max=5.0,
                p50=3.0,
                p90=4.5,
                p95=5.0,
                p99=5.0,
                std=0.8,
            ),
            effective_total_throughput=MetricResult(
                tag="effective_total_throughput",
                header="Effective Total Throughput",
                unit="tokens/sec",
                avg=600.0,
                min=250.0,
                max=1000.0,
                p50=600.0,
                p90=880.0,
                p95=940.0,
                p99=990.0,
                std=130.0,
            ),
            effective_throughput_per_user=MetricResult(
                tag="effective_throughput_per_user",
                header="Effective Throughput Per User",
                unit="tokens/sec/user",
                avg=20.0,
                min=10.0,
                max=40.0,
                p50=20.0,
                p90=36.0,
                p95=38.0,
                p99=40.0,
                std=6.0,
            ),
            effective_prefill_throughput_per_user=MetricResult(
                tag="effective_prefill_throughput_per_user",
                header="Effective Prefill Throughput Per User",
                unit="tokens/sec/user",
                avg=100.0,
                min=40.0,
                max=160.0,
                p50=100.0,
                p90=140.0,
                p95=150.0,
                p99=158.0,
                std=20.0,
            ),
            tokens_in_flight=MetricResult(
                tag="tokens_in_flight",
                header="Tokens In Flight",
                unit="tokens",
                avg=5000.0,
                min=100.0,
                max=12000.0,
                p50=4500.0,
                p90=10000.0,
                p95=11000.0,
                p99=11800.0,
                std=2500.0,
            ),
            window_metadata=SteadyStateWindowMetadata(
                ramp_up_end_ns=100.0,
                ramp_down_start_ns=900.0,
                steady_state_duration_ns=800.0,
                total_requests=10,
                steady_state_requests=8,
                detection_method="cusum_mser5_latency",
                fraction_retained=0.8,
                variance_inflation_factor=1.25,
                effective_p99_sample_size=0,
                sample_size_warning=True,
                trend_correlation=-0.23,
                trend_p_value=0.42,
                stationarity_warning=False,
                cusum_ramp_up_end_ns=90.0,
                cusum_ramp_down_start_ns=910.0,
                mser5_latency_ramp_up_end_ns=100.0,
                mser5_latency_ramp_down_start_ns=900.0,
                mser5_ttft_ramp_up_end_ns=None,
                mser5_ttft_ramp_down_start_ns=None,
                cusum_throughput_ramp_up_end_ns=95.0,
                cusum_throughput_ramp_down_start_ns=905.0,
            ),
        )

    def test_to_json(self) -> None:
        summary = self._make_summary()
        data = summary.to_json()
        assert "results" in data
        assert "window_metadata" in data
        meta = data["window_metadata"]
        assert meta["detection_method"] == "cusum_mser5_latency"
        assert meta["total_requests"] == 10
        # Nested quality group
        quality = meta["quality"]
        assert quality["fraction_retained"] == 0.8
        assert quality["variance_inflation_factor"] == pytest.approx(1.25)
        assert quality["effective_p99_sample_size"] == 0
        assert quality["sample_size_warning"] is True
        # Nested stationarity group
        stationarity = meta["stationarity"]
        assert stationarity["trend_correlation"] == pytest.approx(-0.23)
        assert stationarity["trend_p_value"] == pytest.approx(0.42)
        assert stationarity["stationarity_warning"] is False
        # Nested cross_validation group
        cv = meta["cross_validation"]
        assert cv["cusum_ramp_up_end_ns"] == 90.0
        assert cv["mser5_latency_ramp_up_end_ns"] == 100.0
        assert cv["mser5_ttft_ramp_up_end_ns"] is None
        assert cv["cusum_throughput_ramp_up_end_ns"] == 95.0
        assert cv["cusum_throughput_ramp_down_start_ns"] == 905.0
        assert data["effective_concurrency"]["avg"] == 5.0
        assert data["effective_concurrency"]["p99"] == 8.0
        assert data["effective_throughput"]["avg"] == 100.0
        assert data["effective_throughput"]["unit"] == "tokens/sec"
        assert data["effective_prefill_throughput"]["avg"] == 500.0
        assert data["effective_prefill_throughput"]["unit"] == "tokens/sec"
        assert data["effective_generation_concurrency"]["avg"] == 4.5
        assert data["effective_generation_concurrency"]["unit"] == "requests"
        assert data["effective_prefill_concurrency"]["avg"] == 3.0
        assert data["effective_prefill_concurrency"]["unit"] == "requests"
        assert data["effective_total_throughput"]["avg"] == 600.0
        assert data["effective_total_throughput"]["unit"] == "tokens/sec"
        assert data["effective_throughput_per_user"]["avg"] == 20.0
        assert data["effective_throughput_per_user"]["unit"] == "tokens/sec/user"
        assert data["effective_prefill_throughput_per_user"]["avg"] == 100.0
        assert (
            data["effective_prefill_throughput_per_user"]["unit"] == "tokens/sec/user"
        )
        assert data["tokens_in_flight"]["avg"] == 5000.0
        assert data["tokens_in_flight"]["unit"] == "tokens"

    def test_sweep_metrics_dict(self) -> None:
        """sweep_metrics property includes all 9 sweep metrics."""
        summary = self._make_summary()
        sm = summary.sweep_metrics
        assert "effective_concurrency" in sm
        assert "effective_throughput" in sm
        assert "effective_prefill_throughput" in sm
        assert "effective_generation_concurrency" in sm
        assert "effective_prefill_concurrency" in sm
        assert "effective_total_throughput" in sm
        assert "effective_throughput_per_user" in sm
        assert "effective_prefill_throughput_per_user" in sm
        assert "tokens_in_flight" in sm
        assert len(sm) == 9

    def test_to_csv(self) -> None:
        summary = self._make_summary()
        rows = summary.to_csv()
        assert len(rows) == 1
        assert rows[0]["tag"] == "test_tag"


class TestSteadyStateAnalyzerStationarityWarning:
    @pytest.mark.asyncio
    async def test_trending_data_fires_warning(
        self, mock_metric_registry: Mock, mock_user_config: AIPerfConfig
    ) -> None:
        """Monotonically increasing latency in window → stationarity_warning=True."""
        # Use user_override to control the window, with trending latency inside
        # 100 requests with monotonically increasing latency
        records = [(i, i * 100, (i + 1) * 100, float(i * 10)) for i in range(100)]
        acc = await _build_accumulator_with_records(
            mock_metric_registry, mock_user_config, records
        )
        ctx = _make_summary_ctx(acc)

        # Use full range so all trending data is included
        config = _make_user_config(start_pct=0.0, end_pct=99.9)
        ss = SteadyStateAnalyzer(run=_make_run(config))
        result = await ss.summarize(ctx)

        assert result.window_metadata.stationarity_warning is True
        assert result.window_metadata.trend_correlation is not None
        assert abs(result.window_metadata.trend_correlation) > 0.6
        assert result.window_metadata.trend_p_value is not None
        assert result.window_metadata.trend_p_value < 0.05


class TestSteadyStateAnalyzerSampleQuality:
    @pytest.mark.asyncio
    async def test_variance_inflation_computed(
        self, mock_metric_registry: Mock, mock_user_config: AIPerfConfig
    ) -> None:
        """VIF = total_requests / steady_state_requests."""
        # 10 sequential requests, user override keeps requests [2..7] (6 of 10)
        records = [(i, i * 100, (i + 1) * 100, float(i)) for i in range(10)]
        acc = await _build_accumulator_with_records(
            mock_metric_registry, mock_user_config, records
        )
        ctx = _make_summary_ctx(acc)

        config = _make_user_config(start_pct=20.0, end_pct=80.0)
        ss = SteadyStateAnalyzer(run=_make_run(config))
        result = await ss.summarize(ctx)

        meta = result.window_metadata
        assert meta.variance_inflation_factor == pytest.approx(10 / 6)

    @pytest.mark.asyncio
    async def test_sample_size_warning_fires_for_small_dataset(
        self, mock_metric_registry: Mock, mock_user_config: AIPerfConfig
    ) -> None:
        """50 records with narrow window → effective_p99_sample_size < 10 → warning."""
        records = [(i, i * 100, (i + 1) * 100, float(i)) for i in range(50)]
        acc = await _build_accumulator_with_records(
            mock_metric_registry, mock_user_config, records
        )
        ctx = _make_summary_ctx(acc)

        config = _make_user_config(start_pct=0.0, end_pct=99.9)
        ss = SteadyStateAnalyzer(run=_make_run(config))
        result = await ss.summarize(ctx)

        meta = result.window_metadata
        # 50 * 0.01 = 0, so warning should fire
        assert meta.effective_p99_sample_size < 10
        assert meta.sample_size_warning is True

    @pytest.mark.asyncio
    async def test_sample_size_warning_not_fires_for_large_dataset(
        self, mock_metric_registry: Mock, mock_user_config: AIPerfConfig
    ) -> None:
        """10K+ records → effective_p99_sample_size >= 10 → no warning."""
        records = [(i, i * 10, (i + 1) * 10, float(i % 100)) for i in range(2000)]
        acc = await _build_accumulator_with_records(
            mock_metric_registry, mock_user_config, records
        )
        ctx = _make_summary_ctx(acc)

        config = _make_user_config(start_pct=0.0, end_pct=99.9)
        ss = SteadyStateAnalyzer(run=_make_run(config))
        result = await ss.summarize(ctx)

        meta = result.window_metadata
        # 2000 * 0.01 = 20, so warning should not fire
        assert meta.effective_p99_sample_size >= 10
        assert meta.sample_size_warning is False


# ---------------------------------------------------------------------------
# Effective Throughput Tests
# ---------------------------------------------------------------------------


def _make_throughput_metric():
    """Create metric classes needed for throughput testing."""
    from aiperf.common.enums import MetricType

    class FakeLatency:
        tag = "request_latency"
        type = MetricType.RECORD
        header = "Request Latency"
        unit = "ms"

        def derive_value(self, results):
            raise NotImplementedError

    class FakeOutputTokens:
        tag = "output_tokens"
        type = MetricType.RECORD
        header = "Output Tokens"
        unit = "tokens"

        def derive_value(self, results):
            raise NotImplementedError

    class FakeTTFT:
        tag = "time_to_first_token"
        type = MetricType.RECORD
        header = "Time To First Token"
        unit = "ns"

        def derive_value(self, results):
            raise NotImplementedError

    class FakeISL:
        tag = "input_sequence_length"
        type = MetricType.RECORD
        header = "Input Sequence Length"
        unit = "tokens"

        def derive_value(self, results):
            raise NotImplementedError

    return FakeLatency, FakeOutputTokens, FakeTTFT, FakeISL


async def _build_accumulator_with_throughput_records(
    mock_metric_registry: Mock,
    user_config: AIPerfConfig,
    records: list[tuple[int, int, int, float, float, float]],
    *,
    input_tokens: float | None = None,
) -> object:
    """Build and populate a MetricsAccumulator with throughput-relevant data.

    Args:
        records: list of (session_num, start_ns, end_ns, latency, output_tokens, ttft_ns)
        input_tokens: If set, include input_sequence_length in each record.
    """
    latency_cls, output_cls, ttft_cls, isl_cls = _make_throughput_metric()
    metric_classes = [latency_cls, output_cls, ttft_cls]
    if input_tokens is not None:
        metric_classes.append(isl_cls)
    acc = create_accumulator_with_metrics(user_config, *metric_classes)

    for session_num, start_ns, end_ns, latency, output_tokens_val, ttft_ns in records:
        result: dict = {
            "request_latency": latency,
            "output_tokens": output_tokens_val,
            "time_to_first_token": ttft_ns,
        }
        if input_tokens is not None:
            result["input_sequence_length"] = input_tokens
        msg = create_metric_records_message(
            session_num=session_num,
            request_start_ns=start_ns,
            request_end_ns=end_ns,
            results=[result],
        )
        await acc.process_record(msg.to_data())

    return acc


class TestSteadyStateAnalyzerEffectiveThroughput:
    @pytest.mark.asyncio
    async def test_effective_throughput_present(
        self, mock_metric_registry: Mock, mock_user_config: AIPerfConfig
    ) -> None:
        """effective_throughput field is always present on the result."""
        records = [(i, 0, 1_000_000_000, float(i * 10)) for i in range(50)]
        acc = await _build_accumulator_with_records(
            mock_metric_registry, mock_user_config, records
        )
        ctx = _make_summary_ctx(acc)

        config = _make_user_config()
        ss = SteadyStateAnalyzer(run=_make_run(config))
        result = await ss.summarize(ctx)

        assert hasattr(result, "effective_throughput")
        assert result.effective_throughput.tag == "effective_throughput"
        assert result.effective_throughput.unit == "tokens/sec"

    @pytest.mark.asyncio
    async def test_zero_throughput_without_token_data(
        self, mock_metric_registry: Mock, mock_user_config: AIPerfConfig
    ) -> None:
        """When no output_tokens/TTFT data exists, throughput stats are zero."""
        # Records without output_tokens or time_to_first_token → no generation_start_ns
        records = [(i, i * 100, (i + 1) * 100, float(i)) for i in range(20)]
        acc = await _build_accumulator_with_records(
            mock_metric_registry, mock_user_config, records
        )
        ctx = _make_summary_ctx(acc)

        config = _make_user_config(start_pct=0.0, end_pct=99.9)
        ss = SteadyStateAnalyzer(run=_make_run(config))
        result = await ss.summarize(ctx)

        tput = result.effective_throughput
        assert tput.avg == pytest.approx(0.0)
        assert tput.min == pytest.approx(0.0)
        assert tput.max == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_known_throughput_with_overlapping_requests(
        self, mock_metric_registry: Mock, mock_user_config: AIPerfConfig
    ) -> None:
        """Overlapping requests with known token rates → verify avg throughput."""
        # 10 requests: each starts at 0, ends at 1_000_000_000 (1 sec),
        # TTFT = 0 (generation starts at request start),
        # output_tokens = 101 → per-request rate = 100 tokens / 1e9 ns = 1e-7 tokens/ns
        # 10 overlapping → total ~1000 tokens/sec
        records = [(i, 0, 1_000_000_000, 100.0, 101.0, 0.0) for i in range(10)]
        acc = await _build_accumulator_with_throughput_records(
            mock_metric_registry, mock_user_config, records
        )
        ctx = _make_summary_ctx(acc)

        config = _make_user_config(start_pct=0.0, end_pct=99.9)
        ss = SteadyStateAnalyzer(run=_make_run(config))
        result = await ss.summarize(ctx)

        tput = result.effective_throughput
        # 10 * (101-1) / 1sec = 1000 tokens/sec
        assert tput.avg == pytest.approx(1000.0, rel=0.05)
        assert tput.min >= 0.0
        assert tput.max >= tput.avg


class TestSteadyStateAnalyzerEffectivePrefillThroughput:
    @pytest.mark.asyncio
    async def test_effective_prefill_throughput_present(
        self, mock_metric_registry: Mock, mock_user_config: AIPerfConfig
    ) -> None:
        """effective_prefill_throughput field is always present on the result."""
        records = [(i, 0, 1_000_000_000, float(i * 10)) for i in range(50)]
        acc = await _build_accumulator_with_records(
            mock_metric_registry, mock_user_config, records
        )
        ctx = _make_summary_ctx(acc)

        config = _make_user_config()
        ss = SteadyStateAnalyzer(run=_make_run(config))
        result = await ss.summarize(ctx)

        assert hasattr(result, "effective_prefill_throughput")
        assert result.effective_prefill_throughput.tag == "effective_prefill_throughput"
        assert result.effective_prefill_throughput.unit == "tokens/sec"

    @pytest.mark.asyncio
    async def test_zero_prefill_throughput_without_isl(
        self, mock_metric_registry: Mock, mock_user_config: AIPerfConfig
    ) -> None:
        """When no ISL data exists, prefill throughput stats are zero."""
        records = [(i, i * 100, (i + 1) * 100, float(i)) for i in range(20)]
        acc = await _build_accumulator_with_records(
            mock_metric_registry, mock_user_config, records
        )
        ctx = _make_summary_ctx(acc)

        config = _make_user_config(start_pct=0.0, end_pct=99.9)
        ss = SteadyStateAnalyzer(run=_make_run(config))
        result = await ss.summarize(ctx)

        ptput = result.effective_prefill_throughput
        assert ptput.avg == pytest.approx(0.0)
        assert ptput.min == pytest.approx(0.0)
        assert ptput.max == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_known_prefill_throughput_with_overlapping_requests(
        self, mock_metric_registry: Mock, mock_user_config: AIPerfConfig
    ) -> None:
        """Overlapping requests with known prefill rates → verify avg prefill throughput."""
        # 10 requests: each starts at 0, ends at 1_000_000_000 (1 sec),
        # TTFT = 100_000_000 (100ms), input_tokens = 200
        # Per-request prefill rate = 200 / 100e6 ns = 2e-6 tokens/ns = 2000 tokens/sec
        # 10 overlapping prefills → total ~20000 tokens/sec
        records = [
            (i, 0, 1_000_000_000, 100.0, 101.0, 100_000_000.0) for i in range(10)
        ]
        acc = await _build_accumulator_with_throughput_records(
            mock_metric_registry, mock_user_config, records, input_tokens=200.0
        )
        ctx = _make_summary_ctx(acc)

        config = _make_user_config(start_pct=0.0, end_pct=99.9)
        ss = SteadyStateAnalyzer(run=_make_run(config))
        result = await ss.summarize(ctx)

        ptput = result.effective_prefill_throughput
        # Prefill window is [0, 100ms) for each request
        # During [0, 100ms]: 10 * 200 / 100ms = 20000 tokens/sec
        # But time-weighted over the full window [0, ~1s], the avg will be lower
        # since no prefill activity after 100ms
        assert ptput.avg > 0.0
        assert ptput.max == pytest.approx(20000.0, rel=0.05)


class TestSteadyStateAnalyzerEffectiveThroughputPerUser:
    @pytest.mark.asyncio
    async def test_per_user_throughput_present(
        self, mock_metric_registry: Mock, mock_user_config: AIPerfConfig
    ) -> None:
        """effective_throughput_per_user field is always present on the result."""
        records = [(i, 0, 1_000_000_000, float(i * 10)) for i in range(50)]
        acc = await _build_accumulator_with_records(
            mock_metric_registry, mock_user_config, records
        )
        ctx = _make_summary_ctx(acc)

        config = _make_user_config()
        ss = SteadyStateAnalyzer(run=_make_run(config))
        result = await ss.summarize(ctx)

        assert hasattr(result, "effective_throughput_per_user")
        assert (
            result.effective_throughput_per_user.tag == "effective_throughput_per_user"
        )
        assert result.effective_throughput_per_user.unit == "tokens/sec/user"

    @pytest.mark.asyncio
    async def test_zero_per_user_throughput_without_token_data(
        self, mock_metric_registry: Mock, mock_user_config: AIPerfConfig
    ) -> None:
        """When no output_tokens/TTFT data exists, per-user throughput is zero."""
        records = [(i, i * 100, (i + 1) * 100, float(i)) for i in range(20)]
        acc = await _build_accumulator_with_records(
            mock_metric_registry, mock_user_config, records
        )
        ctx = _make_summary_ctx(acc)

        config = _make_user_config(start_pct=0.0, end_pct=99.9)
        ss = SteadyStateAnalyzer(run=_make_run(config))
        result = await ss.summarize(ctx)

        tput_pu = result.effective_throughput_per_user
        assert tput_pu.avg == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_known_per_user_throughput_single_concurrency(
        self, mock_metric_registry: Mock, mock_user_config: AIPerfConfig
    ) -> None:
        """Single concurrent request → per-user throughput equals aggregate."""
        # 1 request: 101 output tokens over 1 sec = 100 tokens/sec
        records = [(0, 0, 1_000_000_000, 100.0, 101.0, 0.0)]
        acc = await _build_accumulator_with_throughput_records(
            mock_metric_registry, mock_user_config, records
        )
        ctx = _make_summary_ctx(acc)

        config = _make_user_config(start_pct=0.0, end_pct=99.9)
        ss = SteadyStateAnalyzer(run=_make_run(config))
        result = await ss.summarize(ctx)

        tput = result.effective_throughput
        tput_pu = result.effective_throughput_per_user
        # With concurrency 1, per-user ≈ aggregate
        assert tput_pu.avg == pytest.approx(tput.avg, rel=0.05)

    @pytest.mark.asyncio
    async def test_known_per_user_throughput_multiple_concurrent(
        self, mock_metric_registry: Mock, mock_user_config: AIPerfConfig
    ) -> None:
        """10 overlapping requests → per-user throughput ≈ aggregate / 10."""
        # 10 requests: each 101 tokens over 1 sec
        # Aggregate = 1000 tok/sec, per-user = 100 tok/sec
        records = [(i, 0, 1_000_000_000, 100.0, 101.0, 0.0) for i in range(10)]
        acc = await _build_accumulator_with_throughput_records(
            mock_metric_registry, mock_user_config, records
        )
        ctx = _make_summary_ctx(acc)

        config = _make_user_config(start_pct=0.0, end_pct=99.9)
        ss = SteadyStateAnalyzer(run=_make_run(config))
        result = await ss.summarize(ctx)

        tput_pu = result.effective_throughput_per_user
        # Per-user = 100 tokens/sec (1000 / 10)
        assert tput_pu.avg == pytest.approx(100.0, rel=0.05)


class TestSteadyStateAnalyzerEffectivePrefillThroughputPerUser:
    @pytest.mark.asyncio
    async def test_per_user_prefill_throughput_present(
        self, mock_metric_registry: Mock, mock_user_config: AIPerfConfig
    ) -> None:
        """effective_prefill_throughput_per_user field is always present."""
        records = [(i, 0, 1_000_000_000, float(i * 10)) for i in range(50)]
        acc = await _build_accumulator_with_records(
            mock_metric_registry, mock_user_config, records
        )
        ctx = _make_summary_ctx(acc)

        config = _make_user_config()
        ss = SteadyStateAnalyzer(run=_make_run(config))
        result = await ss.summarize(ctx)

        assert hasattr(result, "effective_prefill_throughput_per_user")
        assert (
            result.effective_prefill_throughput_per_user.tag
            == "effective_prefill_throughput_per_user"
        )
        assert result.effective_prefill_throughput_per_user.unit == "tokens/sec/user"

    @pytest.mark.asyncio
    async def test_known_per_user_prefill_throughput_multiple_concurrent(
        self, mock_metric_registry: Mock, mock_user_config: AIPerfConfig
    ) -> None:
        """10 overlapping prefills → per-user prefill ≈ aggregate / 10."""
        # 10 requests: TTFT=100ms, input_tokens=200
        # Per-request: 200/100ms = 2000 tok/sec, aggregate = 20000
        # Per-user = 2000 tok/sec
        records = [
            (i, 0, 1_000_000_000, 100.0, 101.0, 100_000_000.0) for i in range(10)
        ]
        acc = await _build_accumulator_with_throughput_records(
            mock_metric_registry, mock_user_config, records, input_tokens=200.0
        )
        ctx = _make_summary_ctx(acc)

        config = _make_user_config(start_pct=0.0, end_pct=99.9)
        ss = SteadyStateAnalyzer(run=_make_run(config))
        result = await ss.summarize(ctx)

        ptput_pu = result.effective_prefill_throughput_per_user
        # During prefill window, per-user = 2000 tokens/sec
        assert ptput_pu.max == pytest.approx(2000.0, rel=0.05)
