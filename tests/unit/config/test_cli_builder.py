# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for build_aiperf_config() covering all major modes and edge cases."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aiperf.common.enums import AIPerfLogLevel
from aiperf.config.cli_converter import build_aiperf_config
from aiperf.config.cli_model import CLIModel
from aiperf.config.config import AIPerfConfig
from aiperf.plugin.enums import ArrivalPattern, PublicDatasetType, UIType


@pytest.fixture()
def base_kwargs() -> dict:
    """Minimal kwargs to construct a valid CLIModel."""
    return {
        "model_names": ["test-model"],
        "urls": ["http://localhost:8000"],
    }


def _build(base_kwargs: dict, **overrides) -> AIPerfConfig:
    """Build AIPerfConfig from CLIModel with overrides."""
    kwargs = {**base_kwargs, **overrides}
    cli = CLIModel(**kwargs)
    return build_aiperf_config(cli)


class TestConcurrencyMode:
    """Concurrency-only mode (no request rate)."""

    def test_default_concurrency_mode(self, base_kwargs: dict) -> None:
        config = _build(base_kwargs, concurrency=8, request_count=100)
        profiling = config.phases["profiling"]
        assert profiling.type == "concurrency"
        assert profiling.concurrency == 8
        assert profiling.requests == 100

    def test_concurrency_with_duration(self, base_kwargs: dict) -> None:
        config = _build(base_kwargs, concurrency=4, benchmark_duration=60.0)
        profiling = config.phases["profiling"]
        assert profiling.duration == 60.0
        assert profiling.concurrency == 4


class TestRequestRateMode:
    """Request rate mode with various arrival patterns."""

    def test_poisson_rate(self, base_kwargs: dict) -> None:
        config = _build(
            base_kwargs,
            request_rate=10.0,
            arrival_pattern=ArrivalPattern.POISSON,
            request_count=100,
        )
        profiling = config.phases["profiling"]
        assert profiling.type == "poisson"
        assert profiling.rate == 10.0

    def test_constant_rate(self, base_kwargs: dict) -> None:
        config = _build(
            base_kwargs,
            request_rate=5.0,
            arrival_pattern=ArrivalPattern.CONSTANT,
            request_count=50,
        )
        profiling = config.phases["profiling"]
        assert profiling.type == "constant"
        assert profiling.rate == 5.0

    def test_gamma_rate_with_smoothness(self, base_kwargs: dict) -> None:
        config = _build(
            base_kwargs,
            request_rate=20.0,
            arrival_pattern=ArrivalPattern.GAMMA,
            arrival_smoothness=1.5,
            request_count=200,
        )
        profiling = config.phases["profiling"]
        assert profiling.type == "gamma"
        assert profiling.rate == 20.0
        assert profiling.smoothness == 1.5


class TestFixedScheduleMode:
    """Fixed schedule mode."""

    def test_fixed_schedule(self, base_kwargs: dict) -> None:
        config = _build(
            base_kwargs,
            fixed_schedule=True,
            request_count=100,
        )
        profiling = config.phases["profiling"]
        assert profiling.type == "fixed_schedule"


class TestUserCentricMode:
    """User-centric rate mode."""

    def test_user_centric(self, base_kwargs: dict) -> None:
        config = _build(
            base_kwargs,
            user_centric_rate=5.0,
            num_users=10,
            request_count=100,
            num_turns_mean=2,
            num_turns_stddev=0,
        )
        profiling = config.phases["profiling"]
        assert profiling.type == "user_centric"
        assert profiling.rate == 5.0
        assert profiling.users == 10


class TestWarmupPhase:
    """Warmup phase creation."""

    def test_warmup_with_request_count(self, base_kwargs: dict) -> None:
        config = _build(
            base_kwargs,
            warmup_request_count=10,
            concurrency=4,
            request_count=100,
        )
        assert "warmup" in config.phases
        warmup = config.phases["warmup"]
        assert warmup.exclude_from_results is True
        assert warmup.requests == 10

    def test_warmup_with_duration(self, base_kwargs: dict) -> None:
        config = _build(
            base_kwargs,
            warmup_duration=30.0,
            concurrency=4,
            request_count=100,
        )
        warmup = config.phases["warmup"]
        assert warmup.exclude_from_results is True
        assert warmup.duration == 30.0

    def test_warmup_inherits_concurrency(self, base_kwargs: dict) -> None:
        config = _build(
            base_kwargs,
            warmup_request_count=10,
            concurrency=8,
            request_count=100,
        )
        warmup = config.phases["warmup"]
        assert warmup.concurrency == 8

    def test_warmup_overrides_concurrency(self, base_kwargs: dict) -> None:
        config = _build(
            base_kwargs,
            warmup_request_count=10,
            warmup_concurrency=2,
            concurrency=8,
            request_count=100,
        )
        warmup = config.phases["warmup"]
        assert warmup.concurrency == 2

    def test_no_warmup_by_default(self, base_kwargs: dict) -> None:
        config = _build(base_kwargs, concurrency=4, request_count=100)
        assert "warmup" not in config.phases


class TestDatasetInference:
    """Dataset type inference from CLI flags."""

    def test_synthetic_by_default(self, base_kwargs: dict) -> None:
        config = _build(base_kwargs, request_count=100)
        dataset = config.datasets["main"]
        assert dataset.type == "synthetic"

    def test_public_dataset(self, base_kwargs: dict) -> None:
        config = _build(
            base_kwargs,
            public_dataset=PublicDatasetType.SHAREGPT,
            request_count=100,
        )
        dataset = config.datasets["main"]
        assert dataset.type == "public"

    def test_file_dataset(self, base_kwargs: dict, tmp_path: Path) -> None:
        input_file = tmp_path / "data.jsonl"
        input_file.write_text('{"prompt": "test"}\n')
        config = _build(
            base_kwargs,
            input_file=str(input_file),
            request_count=100,
        )
        dataset = config.datasets["main"]
        assert dataset.type == "file"


class TestVerboseFlags:
    """Verbose flag handling."""

    def test_verbose_sets_debug(self, base_kwargs: dict) -> None:
        config = _build(base_kwargs, verbose=True, request_count=100)
        assert config.logging.level == AIPerfLogLevel.DEBUG

    def test_extra_verbose_sets_trace(self, base_kwargs: dict) -> None:
        config = _build(base_kwargs, extra_verbose=True, request_count=100)
        assert config.logging.level == AIPerfLogLevel.TRACE

    def test_verbose_sets_simple_ui(self, base_kwargs: dict) -> None:
        config = _build(base_kwargs, verbose=True, request_count=100)
        assert config.runtime.ui == UIType.SIMPLE


class TestModelsConfig:
    """Model configuration."""

    def test_single_model(self, base_kwargs: dict) -> None:
        config = _build(base_kwargs, request_count=100)
        assert len(config.models.items) == 1
        assert config.models.items[0].name == "test-model"

    def test_multiple_models(self, base_kwargs: dict) -> None:
        config = _build(
            base_kwargs,
            model_names=["model-a", "model-b"],
            request_count=100,
        )
        assert len(config.models.items) == 2
        names = [m.name for m in config.models.items]
        assert "model-a" in names
        assert "model-b" in names


class TestEndpointConfig:
    """Endpoint configuration."""

    def test_streaming(self, base_kwargs: dict) -> None:
        config = _build(base_kwargs, streaming=True, request_count=100)
        assert config.endpoint.streaming is True

    def test_custom_endpoint(self, base_kwargs: dict) -> None:
        config = _build(
            base_kwargs,
            custom_endpoint="/v1/custom",
            request_count=100,
        )
        assert config.endpoint.path == "/v1/custom"

    def test_api_key(self, base_kwargs: dict) -> None:
        config = _build(
            base_kwargs,
            api_key="sk-test-key",
            request_count=100,
        )
        assert config.endpoint.api_key == "sk-test-key"


class TestArtifactsConfig:
    """Artifacts and output configuration."""

    def test_artifact_directory(self, base_kwargs: dict) -> None:
        config = _build(
            base_kwargs,
            artifact_directory=Path("/tmp/test-artifacts"),
            request_count=100,
        )
        assert config.artifacts.dir == Path("/tmp/test-artifacts")

    def test_benchmark_id_generated(self, base_kwargs: dict) -> None:
        config = _build(base_kwargs, request_count=100)
        assert config.artifacts.benchmark_id is not None
        assert len(config.artifacts.benchmark_id) > 0

    def test_checkpoints_directory_and_file(self, base_kwargs: dict) -> None:
        config = _build(base_kwargs, request_count=100)
        assert config.artifacts.checkpoints_dir == config.artifacts.dir / "checkpoints"
        assert (
            config.artifacts.profile_export_partial_json_file
            == config.artifacts.checkpoints_dir / "profile_export_aiperf_partial.json"
        )


class TestGpuTelemetry:
    """GPU telemetry configuration."""

    def test_gpu_telemetry_disabled(self, base_kwargs: dict) -> None:
        config = _build(base_kwargs, no_gpu_telemetry=True, request_count=100)
        assert not config.gpu_telemetry.enabled

    def test_gpu_telemetry_enabled_by_default(self, base_kwargs: dict) -> None:
        config = _build(base_kwargs, request_count=100)
        assert config.gpu_telemetry.enabled


class TestServerMetrics:
    """Server metrics configuration."""

    def test_server_metrics_disabled(self, base_kwargs: dict) -> None:
        config = _build(base_kwargs, no_server_metrics=True, request_count=100)
        assert not config.server_metrics.enabled

    def test_server_metrics_enabled_by_default(self, base_kwargs: dict) -> None:
        config = _build(base_kwargs, request_count=100)
        assert config.server_metrics.enabled


class TestCancellationConfig:
    """Request cancellation configuration."""

    def test_cancellation_rate(self, base_kwargs: dict) -> None:
        config = _build(
            base_kwargs,
            request_cancellation_rate=10.0,
            request_cancellation_delay=1.0,
            concurrency=4,
            request_count=100,
        )
        profiling = config.phases["profiling"]
        assert profiling.cancellation is not None
        assert profiling.cancellation.rate == 10.0
        assert profiling.cancellation.delay == 1.0

    def test_no_cancellation_by_default(self, base_kwargs: dict) -> None:
        config = _build(base_kwargs, concurrency=4, request_count=100)
        profiling = config.phases["profiling"]
        assert profiling.cancellation is None


class TestRampConfig:
    """Ramp configuration for gradual load increase."""

    def test_concurrency_ramp(self, base_kwargs: dict) -> None:
        config = _build(
            base_kwargs,
            concurrency=8,
            concurrency_ramp_duration=10.0,
            request_count=100,
        )
        profiling = config.phases["profiling"]
        assert profiling.concurrency_ramp is not None
        assert profiling.concurrency_ramp.duration == 10.0

    def test_warmup_ramp_falls_back(self, base_kwargs: dict) -> None:
        config = _build(
            base_kwargs,
            warmup_request_count=10,
            concurrency=8,
            concurrency_ramp_duration=10.0,
            request_count=100,
        )
        warmup = config.phases["warmup"]
        assert warmup.concurrency_ramp is not None
        assert warmup.concurrency_ramp.duration == 10.0


class TestCLIModelStructure:
    """Structural integrity tests (replaces test_cli_mapping.py safety net)."""

    EXPECTED_FIELD_COUNT = 143

    def test_field_count(self) -> None:
        assert len(CLIModel.model_fields) == self.EXPECTED_FIELD_COUNT

    def test_all_fields_have_descriptions(self) -> None:
        for name, field_info in CLIModel.model_fields.items():
            assert field_info.description, f"CLIModel.{name} has no description"

    def test_model_instantiates_with_defaults(self) -> None:
        cli = CLIModel()
        assert cli.urls == ["localhost:8000"]


class TestCLIDefaultsMatchConfig:
    """Catch drift between CLIModel help-text defaults and AIPerfConfig actual defaults.

    CLIModel defaults are cosmetic (shown in --help). The _pick_set pattern means
    they never flow into AIPerfConfig. But they should match so --help is accurate.
    """

    @staticmethod
    def _get_config_defaults() -> dict[str, Any]:
        """Build a flat map of CLI field name -> AIPerfConfig default value.

        Covers all 81 CLIModel fields with concrete (non-None) defaults that
        have a config model counterpart. Fields that are CLI-only (no config
        path) are excluded — their defaults are editorial choices for --help.
        """
        from aiperf.config.artifacts import ArtifactsConfig
        from aiperf.config.dataset import (
            AudioConfig,
            ImageConfig,
            PromptConfig,
            RankingsConfig,
            SynthesisConfig,
            VideoConfig,
        )
        from aiperf.config.endpoint import EndpointConfig
        from aiperf.config.models import (
            AccuracyConfig,
            LoggingConfig,
            ModelsAdvanced,
            MultiRunConfig,
            RuntimeConfig,
            TokenizerConfig,
        )
        from aiperf.config.phases import ConcurrencyPhase

        ep = EndpointConfig(urls=["http://localhost:8000"])
        m = ModelsAdvanced(items=[{"name": "x"}])
        tok = TokenizerConfig()
        art = ArtifactsConfig()
        log = LoggingConfig()
        rt = RuntimeConfig()
        mr = MultiRunConfig()
        acc = AccuracyConfig(benchmark="mmlu")
        phase = ConcurrencyPhase(type="concurrency", requests=1)
        prompt = PromptConfig()
        audio = AudioConfig()
        image = ImageConfig()
        video = VideoConfig()
        rankings = RankingsConfig()
        synthesis = SynthesisConfig()

        return {
            # Endpoint
            "model_selection_strategy": m.strategy,
            "url_selection_strategy": ep.url_strategy,
            "endpoint_type": ep.type,
            "streaming": ep.streaming,
            "request_timeout_seconds": ep.timeout,
            "use_legacy_max_tokens": ep.use_legacy_max_tokens,
            "use_server_token_count": ep.use_server_token_count,
            "connection_reuse_strategy": ep.connection_reuse,
            # Tokenizer
            "tokenizer_revision": tok.revision,
            "tokenizer_trust_remote_code": tok.trust_remote_code,
            # Phase defaults
            "concurrency": phase.concurrency,
            # Artifacts
            "artifact_directory": art.dir,
            "export_http_trace": art.trace,
            "show_trace_timing": art.show_trace_timing,
            # Logging / Runtime
            "log_level": log.level,
            "ui_type": rt.ui,
            # Prompts (distributions expose .mean)
            "prompt_batch_size": prompt.batch_size,
            # Audio
            "audio_length_mean": audio.length.mean,
            "audio_batch_size": audio.batch_size,
            "audio_format": audio.format,
            "audio_num_channels": audio.channels,
            # Images
            "image_height_mean": image.height.mean,
            "image_width_mean": image.width.mean,
            "image_batch_size": image.batch_size,
            "image_format": image.format,
            # Video
            "video_batch_size": video.batch_size,
            "video_duration": video.duration,
            "video_fps": video.fps,
            "video_synth_type": video.synth_type,
            "video_format": video.format,
            "video_codec": video.codec,
            "video_audio_sample_rate": video.audio.sample_rate,
            "video_audio_num_channels": video.audio.channels,
            # Rankings
            "passages_mean": rankings.passages.mean,
            "passages_prompt_token_mean": rankings.passage_tokens.mean,
            "query_prompt_token_mean": rankings.query_tokens.mean,
            # Synthesis
            "synthesis_speedup_ratio": synthesis.speedup_ratio,
            "synthesis_prefix_len_multiplier": synthesis.prefix_len_multiplier,
            "synthesis_prefix_root_multiplier": synthesis.prefix_root_multiplier,
            "synthesis_prompt_len_multiplier": synthesis.prompt_len_multiplier,
            # Multi-run
            "num_profile_runs": mr.num_runs,
            "profile_run_cooldown_seconds": mr.cooldown_seconds,
            "confidence_level": mr.confidence_level,
            "profile_run_disable_warmup_after_first": mr.disable_warmup_after_first,
            "set_consistent_seed": mr.set_consistent_seed,
            # Accuracy
            "accuracy_n_shots": acc.n_shots,
            "accuracy_enable_cot": acc.enable_cot,
            "accuracy_verbose": acc.verbose,
        }

    def test_cli_defaults_match_config(self) -> None:
        config_defaults = self._get_config_defaults()
        mismatches = []
        for cli_field, config_default in config_defaults.items():
            cli_default = CLIModel.model_fields[cli_field].default
            if cli_default != config_default:
                mismatches.append(
                    f"  {cli_field}: CLI={cli_default!r} != Config={config_default!r}"
                )
        assert not mismatches, (
            "CLIModel defaults drift from AIPerfConfig:\n" + "\n".join(mismatches)
        )
