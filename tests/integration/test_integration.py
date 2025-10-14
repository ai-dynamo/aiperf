# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive integration test suite for AIPerf."""

from pathlib import Path

import pytest
from pytest import approx

from tests.integration.conftest import AIPerfCLI
from tests.integration.metric_validators import (
    compute_stats,
    extract_metric_values,
    validate_all_metrics,
    validate_metric_stats,
)
from tests.integration.models import FakeAIServer
from tests.integration.utils import (
    create_rankings_dataset,
    extract_base64_video_details,
)

# =============================================================================
# Chat Endpoint - /v1/chat/completions
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestChatEndpoint:
    """Tests for /v1/chat/completions endpoint."""

    async def test_basic_chat(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """Basic non-streaming chat completion."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --ui simple
            """
        )
        assert result.request_count == 10


# =============================================================================
# Completions Endpoint - /v1/completions
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestCompletionsEndpoint:
    """Tests for /v1/completions endpoint."""

    async def test_basic_completions(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """Basic non-streaming completions."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type completions \
                --request-count 10 \
                --concurrency 2 \
                --ui simple
            """
        )
        assert result.request_count == 10


# =============================================================================
# Embeddings Endpoint - /v1/embeddings
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestEmbeddingsEndpoint:
    """Tests for /v1/embeddings endpoint."""

    async def test_basic_embeddings(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """Basic embeddings request."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model text-embedding-3-small \
                --tokenizer gpt2 \
                --url {fakeai_server.url} \
                --endpoint-type embeddings \
                --request-count 10 \
                --concurrency 2 \
                --ui simple
            """
        )
        assert result.request_count == 10
        assert (
            not hasattr(result.json, "time_to_first_token")
            or result.json.time_to_first_token is None
        )


# =============================================================================
# Rankings Endpoint - /v1/ranking
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestRankingsEndpoint:
    """Tests for /v1/ranking endpoint."""

    async def test_basic_rankings(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer, tmp_path: Path
    ):
        """Basic rankings with custom dataset."""
        dataset_path = create_rankings_dataset(tmp_path, 5)

        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type rankings \
                --input-file {dataset_path} \
                --custom-dataset-type single_turn \
                --request-count 10 \
                --concurrency 2 \
                --ui simple
            """
        )
        assert result.request_count == 10


# =============================================================================
# Warmup Phase - Request Lifecycle Management
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestWarmup:
    """Tests for warmup phase functionality."""

    async def test_warmup_phase(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """Warmup requests excluded from profiling metrics."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --warmup-request-count 5 \
                --request-count 15 \
                --concurrency 2 \
                --ui simple
            """
        )
        assert result.request_count == 15

    async def test_warmup_with_streaming(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer
    ):
        """Warmup with streaming enabled."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --streaming \
                --warmup-request-count 10 \
                --request-count 20 \
                --concurrency 4 \
                --ui simple
            """
        )
        assert result.request_count == 20
        assert result.has_streaming_metrics


# =============================================================================
# Streaming Responses - All Endpoints
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestStreaming:
    """Tests for streaming responses across endpoints."""

    async def test_streaming_chat(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """Streaming chat completion with metrics validation."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --streaming \
                --request-count 10 \
                --concurrency 2 \
                --ui simple
            """
        )
        assert result.request_count == 10
        assert result.has_streaming_metrics

    async def test_streaming_completions(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer
    ):
        """Streaming completions with metrics validation."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type completions \
                --streaming \
                --request-count 10 \
                --concurrency 2 \
                --ui simple
            """
        )
        assert result.request_count == 10
        assert result.has_streaming_metrics


# =============================================================================
# Multimodal Inputs - Images, Audio, Video
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestMultimodal:
    """Tests for multimodal inputs (images, audio, video)."""

    async def test_images(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """Chat with image inputs."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --ui simple
            """
        )
        assert result.request_count == 10
        assert result.has_input_images

    async def test_audio(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """Chat with audio inputs."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --audio-length-mean 0.1 \
                --ui simple
            """
        )
        assert result.request_count == 10
        assert result.has_input_audio

    async def test_images_and_audio(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """Chat with combined image and audio inputs."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --audio-length-mean 0.1 \
                --ui simple
            """
        )
        assert result.request_count == 10
        assert result.has_input_images
        assert result.has_input_audio

    async def test_video(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """Video generation with parameter validation."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model-names nvidia/cosmos-reason1-7b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --video-width 512 \
                --video-height 288 \
                --video-duration 5.0 \
                --video-fps 4 \
                --video-synth-type moving_shapes \
                --prompt-input-tokens-mean 50 \
                --num-dataset-entries 1 \
                --request-rate 2.0 \
                --request-count 4
            """
        )
        assert result.request_count == 4
        assert result.has_input_videos

        payload = result.inputs.data[0].payloads[0]
        for message in payload.get("messages", []):
            content = message.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "video_url" in item:
                        video_data = item["video_url"]["url"].split(",")[1]
                        details = extract_base64_video_details(video_data)
                        assert details.width == 512
                        assert details.height == 288
                        assert details.fps == approx(4.0)
                        assert details.duration == approx(5.0)


# =============================================================================
# Media Formats - JPEG, PNG, MP3, WAV Support
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestMediaFormats:
    """Tests for different media format support (JPEG, PNG, MP3, WAV)."""

    @pytest.mark.parametrize("image_format", ["jpeg", "png"])
    async def test_image_formats(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer, image_format: str
    ):
        """Test different image format support."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 5 \
                --concurrency 1 \
                --image-width-mean 128 \
                --image-height-mean 128 \
                --image-format {image_format} \
                --ui simple
            """
        )
        assert result.request_count == 5
        assert result.has_input_images

    @pytest.mark.parametrize("audio_format", ["mp3", "wav"])
    async def test_audio_formats(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer, audio_format: str
    ):
        """Test different audio format support."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 5 \
                --concurrency 1 \
                --audio-length-mean 0.1 \
                --audio-format {audio_format} \
                --ui simple
            """
        )
        assert result.request_count == 5
        assert result.has_input_audio


# =============================================================================
# Deterministic Behavior - Random Seed Control
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestDeterministicBehavior:
    """Tests for deterministic behavior with random seeds."""

    async def test_same_seed_identical_inputs(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer
    ):
        """Same random seed produces identical payloads across runs."""
        result1 = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --random-seed 42 \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --audio-length-mean 0.1 \
                --ui simple
            """
        )

        result2 = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --random-seed 42 \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --audio-length-mean 0.1 \
                --ui simple
            """
        )

        assert result1.request_count == 10
        assert result2.request_count == 10

        inputs_1 = result1.inputs.data
        inputs_2 = result2.inputs.data

        assert len(inputs_1) == len(inputs_2), "Session counts differ"

        for s1, s2 in zip(inputs_1, inputs_2, strict=True):
            assert s1.session_id != s2.session_id
            assert s1.payloads == s2.payloads

    async def test_different_seeds_different_inputs(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer
    ):
        """Different random seeds produce different payloads."""
        result1 = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --random-seed 42 \
                --image-width-mean 128 \
                --image-height-mean 128 \
                --ui simple
            """
        )

        result2 = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --random-seed 123 \
                --image-width-mean 128 \
                --image-height-mean 128 \
                --ui simple
            """
        )

        assert result1.request_count == 10
        assert result2.request_count == 10

        inputs_1 = result1.inputs.data
        inputs_2 = result2.inputs.data

        payloads_different = False
        for s1, s2 in zip(inputs_1, inputs_2, strict=True):
            if s1.payloads != s2.payloads:
                payloads_different = True
                break

        assert payloads_different, "Different seeds should produce different payloads"


# =============================================================================
# GPU Telemetry - DCGM Monitoring
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestGpuTelemetry:
    """Tests for GPU telemetry collection and reporting."""

    async def test_gpu_telemetry(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """GPU telemetry collection with DCGM endpoint."""
        dcgm_url = f"{fakeai_server.url}/dcgm"
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --gpu-telemetry {dcgm_url} \
                --streaming \
                --request-count 100 \
                --concurrency 10 \
                --ui dashboard
            """
        )
        dcgm_url = dcgm_url.replace("http://", "")
        assert result.request_count == 100
        assert result.has_gpu_telemetry
        assert result.json.telemetry_data.endpoints is not None
        assert len(result.json.telemetry_data.endpoints) > 0
        assert result.json.telemetry_data.endpoints[dcgm_url].gpus is not None
        assert len(result.json.telemetry_data.endpoints[dcgm_url].gpus) > 0
        assert (
            result.json.telemetry_data.endpoints[dcgm_url].gpus["gpu_0"].metrics
            is not None
        )
        assert (
            len(result.json.telemetry_data.endpoints[dcgm_url].gpus["gpu_0"].metrics)
            > 0
        )


# =============================================================================
# Request Cancellation - Graceful Interruption Handling
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestRequestCancellation:
    """Tests for request cancellation functionality."""

    async def test_request_cancellation(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer
    ):
        """Request cancellation doesn't break pipeline."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --streaming \
                --request-count 50 \
                --concurrency 5 \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --request-cancellation-rate 0.3 \
                --request-cancellation-delay 0.5 \
                --ui simple
            """,
            timeout=120.0,
        )
        assert result.request_count >= 30


# =============================================================================
# Performance & Stress Testing - High Concurrency Scenarios
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestPerformance:
    """Tests for high concurrency and performance scenarios."""

    async def test_high_concurrency_streaming(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer
    ):
        """High concurrency streaming (100 concurrent requests)."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model Qwen/Qwen3-0.6B \
                --url {fakeai_server.url} \
                --gpu-telemetry {fakeai_server.dcgm_url} \
                --endpoint-type chat \
                --concurrency 100 \
                --request-count 100 \
                --streaming \
                --ui simple
            """
        )
        assert result.request_count == 100
        assert result.has_streaming_metrics

    @pytest.mark.performance
    async def test_high_concurrency_multimodal(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer
    ):
        """Extreme concurrency (1000) with streaming and multimodal inputs."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --gpu-telemetry {fakeai_server.dcgm_url} \
                --endpoint-type chat \
                --streaming \
                --request-count 1000 \
                --concurrency 1000 \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --ui simple
            """,
            timeout=180.0,
        )
        assert result.request_count == 1000
        assert result.has_streaming_metrics

    async def test_high_concurrency_embeddings(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer
    ):
        """High concurrency embeddings (50 concurrent requests)."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model text-embedding-3-small \
                --tokenizer gpt2 \
                --url {fakeai_server.url} \
                --endpoint-type embeddings \
                --concurrency 50 \
                --request-count 200 \
                --ui simple
            """
        )
        assert result.request_count == 200


# =============================================================================
# Output Formats - CSV and JSON Export
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestOutputFormats:
    """Tests for different output export formats."""

    async def test_csv_export(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """CSV export format validation."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --streaming \
                --request-count 10 \
                --concurrency 2 \
                --ui simple
            """
        )
        assert "Metric" in result.csv
        assert "Request Latency" in result.csv

    async def test_json_export(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """JSON export format validation."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --ui simple
            """
        )
        assert result.json is not None
        assert result.json.request_count is not None
        assert result.json.request_latency is not None


# =============================================================================
# UI Options - Different Display Modes
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestUIOptions:
    """Tests for different UI modes."""

    async def test_none_ui(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """None UI mode (no interactive output)."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --ui none
            """
        )
        assert result.request_count == 10
        assert result.has_all_outputs


# =============================================================================
# Dashboard UI - Interactive Dashboard Configurations
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestDashboardUI:
    """Tests for dashboard UI mode with different configurations."""

    async def test_with_request_count(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer
    ):
        """Dashboard with fixed request count."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --ui dashboard \
                --request-count 10 \
                --concurrency 2 \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --audio-length-mean 0.1
            """
        )
        assert result.request_count == 10

    async def test_with_duration(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """Dashboard with time-based limit and streaming."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --ui dashboard \
                --benchmark-duration 10 \
                --streaming \
                --concurrency 3 \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --audio-length-mean 0.1
            """,
            timeout=30.0,
        )
        assert result.request_count >= 3
        assert result.has_streaming_metrics
        assert "Benchmark Duration" in result.csv


# =============================================================================
# Metric Validation - Aggregate vs. Raw Data Consistency
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestMetricValidation:
    """Tests for validating aggregate metrics against raw JSONL data."""

    async def test_validate_single_metric(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer
    ):
        """Validate a single metric's statistics are computed correctly."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --streaming \
                --request-count 50 \
                --concurrency 5 \
                --ui simple
            """
        )

        # Extract request_latency values from JSONL
        latency_values = extract_metric_values(result.jsonl, "request_latency")
        assert len(latency_values) == 50

        # Compute statistics from raw values
        computed = compute_stats(latency_values)

        # Validate against JSON export
        validate_metric_stats(computed, result.json.request_latency, "request_latency")

        # Also validate time_to_first_token for streaming
        ttft_values = extract_metric_values(result.jsonl, "time_to_first_token")
        computed_ttft = compute_stats(ttft_values)
        validate_metric_stats(
            computed_ttft, result.json.time_to_first_token, "time_to_first_token"
        )

    async def test_validate_all_metrics(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer
    ):
        """Validate all metrics in JSON export match computed values from JSONL."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --streaming \
                --request-count 100 \
                --concurrency 10 \
                --ui simple
            """
        )

        # Validate all metrics at once
        computed_metrics = validate_all_metrics(result.jsonl, result.json)

        # Verify we computed statistics for multiple metrics
        assert len(computed_metrics) > 0
        assert "request_latency" in computed_metrics
        assert "time_to_first_token" in computed_metrics
        assert "inter_token_latency" in computed_metrics

        # Check that computed counts match actual record count
        for stats in computed_metrics.values():
            assert stats.count == result.request_count

    async def test_validate_non_streaming_metrics(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer
    ):
        """Validate non-streaming metrics are computed correctly."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 30 \
                --concurrency 3 \
                --ui simple
            """
        )

        # Validate all metrics
        computed_metrics = validate_all_metrics(result.jsonl, result.json)

        # Verify non-streaming specific metrics
        assert "request_latency" in computed_metrics
        assert "output_sequence_length" in computed_metrics
        assert "input_sequence_length" in computed_metrics

        # Ensure streaming metrics don't exist
        assert result.json.time_to_first_token is None
