# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import (
    EnergyMetricUnit,
    FrequencyMetricUnit,
    GenericMetricUnit,
    MetricOverTimeUnit,
    MetricSizeUnit,
    MetricTimeUnit,
    PowerMetricUnit,
    TemperatureMetricUnit,
)
from aiperf.server_metrics.units import (
    _parse_parenthetical_unit,
    _parse_scale_from_description,
    _parse_unit_from_description,
    _parse_unit_from_metric_name,
    infer_unit,
)


class TestParseUnitFromMetricName:
    """Tests for _parse_unit_from_metric_name function."""

    @pytest.mark.parametrize(
        "metric_name,expected_unit",
        [
            # Time units
            ("request_duration_seconds", MetricTimeUnit.SECONDS),
            ("vllm:time_to_first_token_seconds", MetricTimeUnit.SECONDS),
            ("processing_time_milliseconds", MetricTimeUnit.MILLISECONDS),
            ("dynamo_component_nats_service_processing_ms", MetricTimeUnit.MILLISECONDS),
            ("latency_nanoseconds", MetricTimeUnit.NANOSECONDS),
            ("event_time_ns", MetricTimeUnit.NANOSECONDS),
            # Size/data units
            ("response_size_bytes", MetricSizeUnit.BYTES),
            ("memory_kilobytes", MetricSizeUnit.KILOBYTES),
            ("disk_megabytes", MetricSizeUnit.MEGABYTES),
            ("storage_gigabytes", MetricSizeUnit.GIGABYTES),
            # Count/quantity units
            ("server_requests_total", GenericMetricUnit.REQUESTS),
            ("server_error_count", GenericMetricUnit.ERRORS),
            ("vllm:generation_tokens", GenericMetricUnit.TOKENS),
            ("vllm:prompt_tokens", GenericMetricUnit.TOKENS),
            ("dynamo_component_requests", GenericMetricUnit.REQUESTS),
            ("dynamo_component_errors", GenericMetricUnit.ERRORS),
            # Compound suffixes (_X_total -> X, not count)
            ("vllm:iteration_tokens_total", GenericMetricUnit.TOKENS),
            ("dynamo_component_nats_service_requests_total", GenericMetricUnit.REQUESTS),
            ("dynamo_component_nats_service_errors_total", GenericMetricUnit.ERRORS),
            ("dynamo_component_nats_service_processing_ms_total", MetricTimeUnit.MILLISECONDS),
            ("request_duration_seconds_total", MetricTimeUnit.SECONDS),
            ("latency_ns_total", MetricTimeUnit.NANOSECONDS),
            ("network_bytes_total", MetricSizeUnit.BYTES),
            # Ratio/percentage units
            ("memory_ratio", GenericMetricUnit.RATIO),
            ("cache_usage_percent", GenericMetricUnit.PERCENT),
            ("vllm:kv_cache_usage_perc", GenericMetricUnit.PERCENT),
            # Physical units
            ("temperature_celsius", TemperatureMetricUnit.CELSIUS),
            ("power_watts", PowerMetricUnit.WATT),
            ("energy_joules", EnergyMetricUnit.JOULE),
            # Case insensitivity
            ("REQUEST_DURATION_SECONDS", MetricTimeUnit.SECONDS),
            ("Vllm:Kv_Cache_Usage_Perc", GenericMetricUnit.PERCENT),
        ],
    )  # fmt: skip
    def test_parses_known_suffixes(self, metric_name: str, expected_unit):
        """Test that known suffixes are correctly parsed."""
        assert _parse_unit_from_metric_name(metric_name) == expected_unit

    @pytest.mark.parametrize(
        "metric_name",
        [
            "dynamo_frontend_inflight_requests_gauge",  # No known suffix
            "model_context_length",  # _length is not a unit
            "unknown_metric",
            "voltage_volts",  # _volts is not a known suffix
            "vllm:cache_config_info",  # _info is not a known suffix
        ],
    )
    def test_returns_none_for_unknown_suffixes(self, metric_name: str):
        """Test that unknown suffixes return None."""
        assert _parse_unit_from_metric_name(metric_name) is None

    def test_longer_suffix_takes_priority(self):
        """Test that longer suffixes match before shorter ones."""
        # _milliseconds should match before _seconds
        assert (
            _parse_unit_from_metric_name("latency_milliseconds")
            == MetricTimeUnit.MILLISECONDS
        )
        # _nanoseconds should match before _seconds
        assert (
            _parse_unit_from_metric_name("latency_nanoseconds")
            == MetricTimeUnit.NANOSECONDS
        )
        # _tokens_total should match before _total (tokens, not count)
        assert (
            _parse_unit_from_metric_name("iteration_tokens_total")
            == GenericMetricUnit.TOKENS
        )
        # _requests_total should match before _total (requests, not count)
        assert (
            _parse_unit_from_metric_name("http_requests_total")
            == GenericMetricUnit.REQUESTS
        )
        # _errors_total should match before _total (maps to errors)
        assert (
            _parse_unit_from_metric_name("server_errors_total")
            == GenericMetricUnit.ERRORS
        )
        # _ms_total should match before _total (milliseconds, not count)
        assert (
            _parse_unit_from_metric_name("processing_ms_total")
            == MetricTimeUnit.MILLISECONDS
        )
        # _seconds_total should match before _total (seconds, not count)
        assert (
            _parse_unit_from_metric_name("duration_seconds_total")
            == MetricTimeUnit.SECONDS
        )

    @pytest.mark.parametrize(
        "metric_name,expected_unit",
        [
            # New suffixes: _reqs shorthand for requests
            ("sglang:num_running_reqs", GenericMetricUnit.REQUESTS),
            ("num_waiting_reqs", GenericMetricUnit.REQUESTS),
            ("queue_reqs", GenericMetricUnit.REQUESTS),
            # New suffixes: _gb_s for throughput in GB/s
            ("sglang:cache_transfer_gb_s", MetricOverTimeUnit.GB_PER_SECOND),
            ("memory_bandwidth_gb_s", MetricOverTimeUnit.GB_PER_SECOND),
        ],
    )
    def test_parses_new_suffixes(self, metric_name: str, expected_unit):
        """Test newly added suffixes."""
        assert _parse_unit_from_metric_name(metric_name) == expected_unit

    @pytest.mark.parametrize(
        "metric_name,expected_unit",
        [
            # *num_requests_* wildcard pattern
            ("vllm:num_requests_running", GenericMetricUnit.REQUESTS),
            ("vllm:num_requests_waiting", GenericMetricUnit.REQUESTS),
            ("num_requests_active", GenericMetricUnit.REQUESTS),
            ("NUM_REQUESTS_PENDING", GenericMetricUnit.REQUESTS),  # case insensitive
            # *request_success wildcard pattern
            ("vllm:request_success", GenericMetricUnit.REQUESTS),
            ("trtllm:request_success", GenericMetricUnit.REQUESTS),
            ("request_success", GenericMetricUnit.REQUESTS),
        ],
    )
    def test_parses_wildcard_patterns(self, metric_name: str, expected_unit):
        """Test glob-style wildcard pattern matching for metric names."""
        assert _parse_unit_from_metric_name(metric_name) == expected_unit


class TestParseParentheticalUnit:
    """Tests for _parse_parenthetical_unit function (generic "(in <unit>)" parser)."""

    @pytest.mark.parametrize(
        "description,expected",
        [
            # DCGM-style metrics (primary use case)
            ("Framebuffer memory free (in MiB).", MetricSizeUnit.MEGABYTES),
            ("Framebuffer memory reserved (in MiB).", MetricSizeUnit.MEGABYTES),
            ("Framebuffer memory used (in MiB).", MetricSizeUnit.MEGABYTES),
            ("GPU temperature (in C).", TemperatureMetricUnit.CELSIUS),
            ("Memory temperature (in C).", TemperatureMetricUnit.CELSIUS),
            ("Memory clock frequency (in MHz).", FrequencyMetricUnit.MEGAHERTZ),
            ("Power draw (in W).", PowerMetricUnit.WATT),
            ("SM clock frequency (in MHz).", FrequencyMetricUnit.MEGAHERTZ),
            (
                "Total energy consumption since boot (in mJ).",
                EnergyMetricUnit.MILLIJOULE,
            ),
            # Other units
            ("Storage capacity (in GB).", MetricSizeUnit.GIGABYTES),
            ("Temperature reading (in °C).", TemperatureMetricUnit.CELSIUS),
            ("CPU frequency (in GHz).", FrequencyMetricUnit.GIGAHERTZ),
            ("Battery charge (in mW).", PowerMetricUnit.MILLIWATT),
            ("Total energy (in J).", EnergyMetricUnit.JOULE),
            ("Large energy consumption (in MJ).", EnergyMetricUnit.MEGAJOULE),
            ("Latency (in ms).", MetricTimeUnit.MILLISECONDS),
            ("Time elapsed (in s).", MetricTimeUnit.SECONDS),
            ("Bandwidth (in GB/s).", MetricOverTimeUnit.GB_PER_SECOND),
            # Percentage
            ("GPU utilization (in %).", GenericMetricUnit.PERCENT),
            ("Memory utilization (in percent).", GenericMetricUnit.PERCENT),
            # Case variations - no match (case-sensitive only to avoid mJ/MJ ambiguity)
            ("Some metric (in mib).", None),  # lowercase doesn't match MiB
            ("Some metric (in MHZ).", None),  # uppercase doesn't match MHz
            # No match cases
            ("Some metric without units", None),
            ("", None),
            (None, None),
            # Edge cases: other parenthetical patterns that shouldn't match
            ("Cache hit rate (0.0-1.0)", None),  # Range pattern, not unit
            ("Memory (bytes)", None),  # No "in" keyword
        ],
    )  # fmt: skip
    def test_parses_parenthetical_units(self, description: str | None, expected):
        """Test that (in <unit>) patterns are correctly parsed."""
        assert _parse_parenthetical_unit(description) == expected

    def test_case_sensitive_for_millijoule_vs_megajoule(self):
        """Test that mJ (millijoule) and MJ (megajoule) are distinguished."""
        assert (
            _parse_parenthetical_unit("Energy (in mJ).") == EnergyMetricUnit.MILLIJOULE
        )
        assert (
            _parse_parenthetical_unit("Energy (in MJ).") == EnergyMetricUnit.MEGAJOULE
        )


class TestParseScaleFromDescription:
    """Tests for _parse_scale_from_description function."""

    @pytest.mark.parametrize(
        "description,expected",
        [
            # Ratio patterns: 0-1 range
            ("GPU cache usage as a percentage (0.0-1.0)", GenericMetricUnit.RATIO),
            ("Cache hit rate (0.0 - 1.0)", GenericMetricUnit.RATIO),
            ("Utilization ratio (0-1)", GenericMetricUnit.RATIO),
            ("Value in range 0.0 to 1.0", GenericMetricUnit.RATIO),
            ("Metric range 0-1", GenericMetricUnit.RATIO),
            ("(0.0–1.0)", GenericMetricUnit.RATIO),  # en-dash
            ("(0.0—1.0)", GenericMetricUnit.RATIO),  # em-dash
            # Ratio patterns: "1 means/is/equals 100" style
            ("KV-cache usage. 1 means 100 percent usage.", GenericMetricUnit.RATIO),
            ("Utilization where 1 means 100%", GenericMetricUnit.RATIO),
            ("Cache fill level (1 = 100%)", GenericMetricUnit.RATIO),
            ("1.0 = 100% full", GenericMetricUnit.RATIO),
            ("Value where 1 is 100%", GenericMetricUnit.RATIO),
            ("1 == 100 percent", GenericMetricUnit.RATIO),
            ("1 equals 100%", GenericMetricUnit.RATIO),
            # Percent patterns: 0-100 range
            ("Memory usage (0-100)", GenericMetricUnit.PERCENT),
            ("CPU utilization (0.0-100.0)", GenericMetricUnit.PERCENT),
            ("Value 0-100%", GenericMetricUnit.PERCENT),
            ("Range 0 to 100", GenericMetricUnit.PERCENT),
            # No range indicator
            ("Current number of running requests", None),
            ("Tokens processed per second", None),
            ("", None),
            (None, None),
        ],
    )  # fmt: skip
    def test_detects_scale_from_description(self, description: str | None, expected):
        assert _parse_scale_from_description(description) == expected

    def test_ratio_takes_priority_over_suffix_naming(self):
        """Test that descriptions with (0.0-1.0) return ratio even if 'percent' appears."""
        # This is the key case: metric named "_percent" but actually 0-1 range
        desc = "GPU cache usage as a percentage (0.0-1.0)"
        assert _parse_scale_from_description(desc) == GenericMetricUnit.RATIO


class TestParseUnitFromDescription:
    """Tests for _parse_unit_from_description function."""

    @pytest.mark.parametrize(
        "description,expected",
        [
            # Explicit "in X" patterns
            ("Request latency in seconds", MetricTimeUnit.SECONDS),
            ("Duration measured in milliseconds", MetricTimeUnit.MILLISECONDS),
            ("Time in ms", MetricTimeUnit.MILLISECONDS),
            ("Interval in nanoseconds", MetricTimeUnit.NANOSECONDS),
            ("Delay in ns", MetricTimeUnit.NANOSECONDS),
            ("Response size in bytes", MetricSizeUnit.BYTES),
            ("Throughput in GB/s", MetricOverTimeUnit.GB_PER_SECOND),
            ("Bandwidth in MB/s", MetricOverTimeUnit.MB_PER_SECOND),
            ("Generation rate in tokens/s", MetricOverTimeUnit.TOKENS_PER_SECOND),
            ("Rate in tokens/sec", MetricOverTimeUnit.TOKENS_PER_SECOND),
            ("Rate in tokens/second", MetricOverTimeUnit.TOKENS_PER_SECOND),
            ("Serving rate in requests/s", MetricOverTimeUnit.REQUESTS_PER_SECOND),
            # Parenthetical patterns
            ("Request latency (seconds)", MetricTimeUnit.SECONDS),
            ("Processing time (milliseconds)", MetricTimeUnit.MILLISECONDS),
            ("Delay (ms)", MetricTimeUnit.MILLISECONDS),
            ("Event time (nanoseconds)", MetricTimeUnit.NANOSECONDS),
            ("Interval (ns)", MetricTimeUnit.NANOSECONDS),
            ("Size (bytes)", MetricSizeUnit.BYTES),
            ("Throughput (GB/s)", MetricOverTimeUnit.GB_PER_SECOND),
            ("Bandwidth (MB/s)", MetricOverTimeUnit.MB_PER_SECOND),
            ("Rate (tokens/s)", MetricOverTimeUnit.TOKENS_PER_SECOND),
            ("Rate (requests/s)", MetricOverTimeUnit.REQUESTS_PER_SECOND),
            # Case insensitivity
            ("Request latency IN SECONDS", MetricTimeUnit.SECONDS),
            ("Rate IN TOKENS/S", MetricOverTimeUnit.TOKENS_PER_SECOND),
            # No unit
            ("Current queue depth", None),
            ("Number of active workers", None),
            ("", None),
            (None, None),
        ],
    )  # fmt: skip
    def test_extracts_unit_from_description(self, description: str | None, expected):
        assert _parse_unit_from_description(description) == expected


class TestInferUnit:
    """Tests for the combined infer_unit function."""

    def test_scale_takes_priority_over_all(self):
        """Scale from description should override everything else."""
        # Even with _percent suffix and existing_unit="percent", (0.0-1.0) means ratio
        result = infer_unit(
            metric_name="gpu_cache_usage_percent",
            description="GPU cache usage as a percentage (0.0-1.0)",
        )
        assert result == GenericMetricUnit.RATIO

    def test_description_unit_over_existing_and_suffix(self):
        """Unit from description should override existing_unit and suffix."""
        result = infer_unit(
            metric_name="some_metric_total",
            description="Value in seconds",
        )
        assert result == MetricTimeUnit.SECONDS

    def test_suffix_fallback_when_no_description_unit(self):
        """Suffix-based inference is used when description has no unit."""
        result = infer_unit(
            metric_name="some_metric_total",
            description="Some generic description",
        )
        assert result == GenericMetricUnit.COUNT

    def test_suffix_as_fallback(self):
        """Suffix-based inference when no other source provides unit."""
        result = infer_unit(
            metric_name="request_duration_seconds",
            description=None,
        )
        assert result == MetricTimeUnit.SECONDS

    def test_none_when_no_unit_found(self):
        """Return None when no unit can be inferred."""
        result = infer_unit(
            metric_name="some_unknown_metric",
            description="A metric with no unit info",
        )
        assert result is None

    @pytest.mark.parametrize(
        "metric_name,description,expected",
        [
            # Real-world examples from SGLang/TensorRT-LLM
            ("sglang:gen_throughput", "Generation throughput in tokens/s", MetricOverTimeUnit.TOKENS_PER_SECOND),
            ("trtllm:e2e_request_latency_seconds", "End-to-end request latency", MetricTimeUnit.SECONDS),
            ("dynamo_component_kvstats_gpu_cache_usage_percent", "GPU cache usage as a percentage (0.0-1.0)", GenericMetricUnit.RATIO),
            ("sglang:cache_hit_rate", "Cache hit rate (0.0-1.0)", GenericMetricUnit.RATIO),
            ("sglang:num_running_reqs", "Number of running requests", GenericMetricUnit.REQUESTS),
            ("vllm:iteration_tokens_total", "Total tokens processed", GenericMetricUnit.TOKENS),
        ],
    )  # fmt: skip
    def test_real_world_metrics(
        self,
        metric_name: str,
        description: str,
        expected,
    ):
        """Test with real-world metric examples."""
        result = infer_unit(metric_name, description)
        assert result == expected


# All unique metrics from concurrency50 (vLLM), concurrency51 (SGLang), concurrency52 (TensorRT-LLM)
# Format: (metric_name, description, expected_unit)
# expected_unit is what the unit SHOULD be based on metric semantics
REAL_WORLD_METRICS_FROM_EXPORTS = [
    # Dynamo metrics
    ("dynamo_component_errors", "Total number of errors in work handler processing", GenericMetricUnit.ERRORS),  # _errors -> ERRORS
    ("dynamo_component_inflight_requests", "Number of requests currently being processed by work handler", GenericMetricUnit.REQUESTS),
    ("dynamo_component_kvstats_active_blocks", "Number of active KV cache blocks currently in use", GenericMetricUnit.BLOCKS),
    ("dynamo_component_kvstats_gpu_cache_usage_percent", "GPU cache usage as a percentage (0.0-1.0)", GenericMetricUnit.RATIO),
    ("dynamo_component_kvstats_gpu_prefix_cache_hit_rate", "GPU prefix cache hit rate as a percentage (0.0-1.0)", GenericMetricUnit.RATIO),
    ("dynamo_component_kvstats_total_blocks", "Total number of KV cache blocks available", GenericMetricUnit.BLOCKS),
    ("dynamo_component_nats_client_connection_state", "Current connection state of NATS client (0=disconnected, 1=connected, 2=reconnecting)", None),
    ("dynamo_component_nats_client_current_connections", "Current number of active connections for NATS client", None),
    ("dynamo_component_nats_client_in_messages", "Total number of messages received by NATS client", None),
    ("dynamo_component_nats_client_in_total_bytes", "Total number of bytes received by NATS client", MetricSizeUnit.BYTES),
    ("dynamo_component_nats_client_out_messages", "Total number of messages sent by NATS client", None),
    ("dynamo_component_nats_client_out_overhead_bytes", "Total number of bytes sent by NATS client", MetricSizeUnit.BYTES),
    ("dynamo_component_nats_service_active_endpoints", "Number of active endpoints across all services", None),
    ("dynamo_component_nats_service_active_services", "Number of active services in this component", None),
    ("dynamo_component_nats_service_errors_total", "Total number of errors across all component endpoints", GenericMetricUnit.ERRORS),  # _errors_total -> ERRORS
    ("dynamo_component_nats_service_processing_ms_avg", "Average processing time across all component endpoints in milliseconds", MetricTimeUnit.MILLISECONDS),
    ("dynamo_component_nats_service_processing_ms_total", "Total processing time across all component endpoints in milliseconds", MetricTimeUnit.MILLISECONDS),
    ("dynamo_component_nats_service_requests_total", "Total number of requests across all component endpoints", GenericMetricUnit.REQUESTS),
    ("dynamo_component_request_bytes", "Total number of bytes received in requests by work handler", MetricSizeUnit.BYTES),
    ("dynamo_component_request_duration_seconds", "Time spent processing requests by work handler", MetricTimeUnit.SECONDS),
    ("dynamo_component_requests", "Total number of requests processed by work handler", GenericMetricUnit.REQUESTS),
    ("dynamo_component_response_bytes", "Total number of bytes sent in responses by work handler", MetricSizeUnit.BYTES),
    ("dynamo_component_uptime_seconds", "Total uptime of the DistributedRuntime in seconds", MetricTimeUnit.SECONDS),
    ("dynamo_frontend_disconnected_clients", "Number of disconnected clients", None),
    ("dynamo_frontend_inflight_requests", "Number of inflight requests", GenericMetricUnit.REQUESTS),
    ("dynamo_frontend_input_sequence_tokens", "Input sequence length in tokens", GenericMetricUnit.TOKENS),
    ("dynamo_frontend_inter_token_latency_seconds", "Inter-token latency in seconds", MetricTimeUnit.SECONDS),
    ("dynamo_frontend_model_context_length", "Maximum context length in tokens for a worker serving the model", GenericMetricUnit.TOKENS),
    ("dynamo_frontend_model_kv_cache_block_size", "KV cache block size in tokens for a worker serving the model", GenericMetricUnit.TOKENS),
    ("dynamo_frontend_model_max_num_batched_tokens", "Maximum number of batched tokens for a worker serving the model", GenericMetricUnit.TOKENS),
    ("dynamo_frontend_model_max_num_seqs", "Maximum number of sequences for a worker serving the model", None),
    ("dynamo_frontend_model_migration_limit", "Maximum number of request migrations allowed for the model", None),
    ("dynamo_frontend_model_total_kv_blocks", "Total KV cache blocks available for a worker serving the model", GenericMetricUnit.BLOCKS),
    ("dynamo_frontend_output_sequence_tokens", "Output sequence length in tokens", GenericMetricUnit.TOKENS),
    ("dynamo_frontend_output_tokens", "Total number of output tokens generated (updates in real-time)", GenericMetricUnit.TOKENS),
    ("dynamo_frontend_queued_requests", "Number of requests in HTTP processing queue", GenericMetricUnit.REQUESTS),
    ("dynamo_frontend_request_duration_seconds", "Duration of LLM requests", MetricTimeUnit.SECONDS),
    ("dynamo_frontend_requests", "Total number of LLM requests processed", GenericMetricUnit.REQUESTS),
    ("dynamo_frontend_time_to_first_token_seconds", "Time to first token in seconds", MetricTimeUnit.SECONDS),
    # SGLang metrics
    ("sglang:cache_hit_rate", "The prefix cache hit rate.", None),  # Ambiguous: _rate could be ratio or per-second
    ("sglang:engine_load_weights_time", "The time taken for the engine to load weights.", None),  # No unit specified
    ("sglang:engine_startup_time", "The time taken for the engine to start up.", None),  # No unit specified
    ("sglang:gen_throughput", "The generation throughput (token/s).", MetricOverTimeUnit.TOKENS_PER_SECOND),
    ("sglang:is_cuda_graph", "Whether the batch is using CUDA graph.", None),  # Boolean flag
    ("sglang:kv_transfer_alloc_ms", "The allocation waiting time of the KV transfer in ms.", MetricTimeUnit.MILLISECONDS),
    ("sglang:kv_transfer_bootstrap_ms", "The bootstrap time of the KV transfer in ms.", MetricTimeUnit.MILLISECONDS),
    ("sglang:kv_transfer_latency_ms", "The transfer latency of the KV cache in ms.", MetricTimeUnit.MILLISECONDS),
    ("sglang:kv_transfer_speed_gb_s", "The transfer speed of the KV cache in GB/s.", MetricOverTimeUnit.GB_PER_SECOND),
    ("sglang:mamba_usage", "The token usage for Mamba layers.", None),  # Ambiguous: usage could be count or ratio
    ("sglang:num_decode_prealloc_queue_reqs", "The number of requests in the decode prealloc queue.", GenericMetricUnit.REQUESTS),
    ("sglang:num_decode_transfer_queue_reqs", "The number of requests in the decode transfer queue.", GenericMetricUnit.REQUESTS),
    ("sglang:num_grammar_queue_reqs", "The number of requests in the grammar waiting queue.", GenericMetricUnit.REQUESTS),
    ("sglang:num_paused_reqs", "The number of paused requests by async weight sync.", GenericMetricUnit.REQUESTS),
    ("sglang:num_prefill_inflight_queue_reqs", "The number of requests in the prefill inflight queue.", GenericMetricUnit.REQUESTS),
    ("sglang:num_prefill_prealloc_queue_reqs", "The number of requests in the prefill prealloc queue.", GenericMetricUnit.REQUESTS),
    ("sglang:num_queue_reqs", "The number of requests in the waiting queue.", GenericMetricUnit.REQUESTS),
    ("sglang:num_retracted_reqs", "The number of retracted requests.", GenericMetricUnit.REQUESTS),
    ("sglang:num_running_reqs", "The number of running requests.", GenericMetricUnit.REQUESTS),
    ("sglang:num_running_reqs_offline_batch", "The number of running low-priority offline batch requests(label is 'batch').", None),  # No _reqs suffix
    ("sglang:num_used_tokens", "The number of used tokens.", GenericMetricUnit.TOKENS),
    ("sglang:pending_prealloc_token_usage", "The token usage for pending preallocated tokens (not preallocated yet).", None),  # Ambiguous
    ("sglang:per_stage_req_latency_seconds", "The latency of each stage of requests.", MetricTimeUnit.SECONDS),
    ("sglang:queue_time_seconds", "Histogram of queueing time in seconds.", MetricTimeUnit.SECONDS),
    ("sglang:spec_accept_length", "The average acceptance length of speculative decoding.", None),  # Ambiguous: length of what?
    ("sglang:spec_accept_rate", "The average acceptance rate of speculative decoding (`accepted tokens / total draft tokens` in batch).", None),  # Ambiguous: rate could mean ratio or per-second
    ("sglang:swa_token_usage", "The token usage for SWA layers.", None),  # Ambiguous
    ("sglang:token_usage", "The token usage.", None),  # Ambiguous
    ("sglang:utilization", "The utilization.", None),  # Ambiguous: no range specified
    # TensorRT-LLM metrics
    ("trtllm:e2e_request_latency_seconds", "Histogram of end to end request latency in seconds.", MetricTimeUnit.SECONDS),
    ("trtllm:request_queue_time_seconds", "Histogram of time spent in WAITING phase for request.", MetricTimeUnit.SECONDS),
    ("trtllm:request_success", "Count of successfully processed requests.", GenericMetricUnit.REQUESTS),
    ("trtllm:time_per_output_token_seconds", "Histogram of time per output token in seconds.", MetricTimeUnit.SECONDS),
    ("trtllm:time_to_first_token_seconds", "Histogram of time to first token in seconds.", MetricTimeUnit.SECONDS),
    # vLLM metrics
    ("vllm:cache_config_info", "Information of the LLMEngine CacheConfig", None),  # _info is not a known suffix
    ("vllm:e2e_request_latency_seconds", "Histogram of e2e request latency in seconds.", MetricTimeUnit.SECONDS),
    ("vllm:generation_tokens", "Number of generation tokens processed.", GenericMetricUnit.TOKENS),
    ("vllm:inter_token_latency_seconds", "Histogram of inter-token latency in seconds.", MetricTimeUnit.SECONDS),
    ("vllm:iteration_tokens_total", "Histogram of number of tokens per engine_step.", GenericMetricUnit.TOKENS),
    ("vllm:kv_cache_usage_perc", "KV-cache usage. 1 means 100 percent usage.", GenericMetricUnit.RATIO),  # "1 means 100 percent" overrides _perc suffix
    ("vllm:num_preemptions", "Cumulative number of preemption from the engine.", None),
    ("vllm:num_requests_running", "Number of requests in model execution batches.", GenericMetricUnit.REQUESTS),
    ("vllm:num_requests_waiting", "Number of requests waiting to be processed.", GenericMetricUnit.REQUESTS),
    ("vllm:prefix_cache_hits", "Prefix cache hits, in terms of number of cached tokens.", None),
    ("vllm:prefix_cache_queries", "Prefix cache queries, in terms of number of queried tokens.", None),
    ("vllm:prompt_tokens", "Number of prefill tokens processed.", GenericMetricUnit.TOKENS),
    ("vllm:request_decode_time_seconds", "Histogram of time spent in DECODE phase for request.", MetricTimeUnit.SECONDS),
    ("vllm:request_generation_tokens", "Number of generation tokens processed.", GenericMetricUnit.TOKENS),
    ("vllm:request_inference_time_seconds", "Histogram of time spent in RUNNING phase for request.", MetricTimeUnit.SECONDS),
    ("vllm:request_max_num_generation_tokens", "Histogram of maximum number of requested generation tokens.", GenericMetricUnit.TOKENS),
    ("vllm:request_params_max_tokens", "Histogram of the max_tokens request parameter.", GenericMetricUnit.TOKENS),
    ("vllm:request_params_n", "Histogram of the n request parameter.", None),
    ("vllm:request_prefill_time_seconds", "Histogram of time spent in PREFILL phase for request.", MetricTimeUnit.SECONDS),
    ("vllm:request_prompt_tokens", "Number of prefill tokens processed.", GenericMetricUnit.TOKENS),
    ("vllm:request_queue_time_seconds", "Histogram of time spent in WAITING phase for request.", MetricTimeUnit.SECONDS),
    ("vllm:request_success", "Count of successfully processed requests.", GenericMetricUnit.REQUESTS),
    ("vllm:request_time_per_output_token_seconds", "Histogram of time_per_output_token_seconds per request.", MetricTimeUnit.SECONDS),
    ("vllm:time_per_output_token_seconds", "Histogram of time per output token in seconds.DEPRECATED: Use vllm:inter_token_latency_seconds instead.", MetricTimeUnit.SECONDS),
    ("vllm:time_to_first_token_seconds", "Histogram of time to first token in seconds.", MetricTimeUnit.SECONDS),
    # DCGM metrics (GPU telemetry)
    ("DCGM_FI_DEV_FB_FREE", "Framebuffer memory free (in MiB).", MetricSizeUnit.MEGABYTES),
    ("DCGM_FI_DEV_FB_RESERVED", "Framebuffer memory reserved (in MiB).", MetricSizeUnit.MEGABYTES),
    ("DCGM_FI_DEV_FB_USED", "Framebuffer memory used (in MiB).", MetricSizeUnit.MEGABYTES),
    ("DCGM_FI_DEV_GPU_TEMP", "GPU temperature (in C).", TemperatureMetricUnit.CELSIUS),
    ("DCGM_FI_DEV_GPU_UTIL", "GPU utilization (in %).", GenericMetricUnit.PERCENT),
    ("DCGM_FI_DEV_MEMORY_TEMP", "Memory temperature (in C).", TemperatureMetricUnit.CELSIUS),
    ("DCGM_FI_DEV_MEM_CLOCK", "Memory clock frequency (in MHz).", FrequencyMetricUnit.MEGAHERTZ),
    ("DCGM_FI_DEV_MEM_COPY_UTIL", "Memory utilization (in %).", GenericMetricUnit.PERCENT),
    ("DCGM_FI_DEV_POWER_USAGE", "Power draw (in W).", PowerMetricUnit.WATT),
    ("DCGM_FI_DEV_SM_CLOCK", "SM clock frequency (in MHz).", FrequencyMetricUnit.MEGAHERTZ),
    ("DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION", "Total energy consumption since boot (in mJ).", EnergyMetricUnit.MILLIJOULE),
]  # fmt: skip


class TestInferUnitRealWorldMetrics:
    """Test infer_unit with all real-world metrics from vLLM, SGLang, and TensorRT-LLM exports.

    These are the EXPECTED units based on metric semantics - what the unit SHOULD be.
    The infer_unit function should be updated to match these expectations.
    """

    @pytest.mark.parametrize(
        "metric_name,description,expected",
        REAL_WORLD_METRICS_FROM_EXPORTS,
        ids=[m[0] for m in REAL_WORLD_METRICS_FROM_EXPORTS],
    )
    def test_infer_unit_real_world(
        self,
        metric_name: str,
        description: str,
        expected,
    ):
        """Test unit inference for real metrics from server exports."""
        result = infer_unit(metric_name, description)
        assert result == expected
