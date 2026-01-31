# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
from pydantic import ValidationError

from aiperf.common.config import (
    ConversationConfig,
    EndpointConfig,
    EndpointDefaults,
    InputConfig,
    LoadGeneratorConfig,
    OutputConfig,
    PromptConfig,
    RankingsConfig,
    RankingsPassagesConfig,
    RankingsQueryConfig,
    TokenizerConfig,
    TurnConfig,
    TurnDelayConfig,
    UserConfig,
)
from aiperf.common.config.prompt_config import InputTokensConfig
from aiperf.common.enums import GPUTelemetryMode
from aiperf.plugin.enums import (
    ArrivalPattern,
    DatasetSamplingStrategy,
    EndpointType,
    TimingMode,
)

# =============================================================================
# Test Helpers
# =============================================================================


def make_endpoint(
    endpoint_type: EndpointType = EndpointType.CHAT,
    streaming: bool = False,
    model_names: list[str] | None = None,
    **kwargs,
) -> EndpointConfig:
    """Create an EndpointConfig with sensible defaults for testing."""
    # Convert url= to urls= for backward compatibility
    if "url" in kwargs:
        kwargs["urls"] = [kwargs.pop("url")]
    return EndpointConfig(
        model_names=model_names or ["test-model"],
        type=endpoint_type,
        custom_endpoint=kwargs.pop("custom_endpoint", "test"),
        streaming=streaming,
        **kwargs,
    )


def make_config(
    endpoint: EndpointConfig | None = None,
    loadgen: LoadGeneratorConfig | None = None,
    input_config: InputConfig | None = None,
    **kwargs,
) -> UserConfig:
    """Create a UserConfig with sensible defaults for testing."""
    config_kwargs = {"endpoint": endpoint or make_endpoint(), **kwargs}
    if loadgen is not None:
        config_kwargs["loadgen"] = loadgen
    if input_config is not None:
        config_kwargs["input"] = input_config
    return UserConfig(**config_kwargs)


def make_multi_turn_input(num: int | None = None, turn_mean: int = 2) -> InputConfig:
    """Create an InputConfig configured for multi-turn conversations.

    Args:
        num: Number of sessions (conversations). If None, not set.
        turn_mean: Mean number of turns per session. Defaults to 2.

    Returns:
        InputConfig with multi-turn conversation settings.
    """
    return InputConfig(
        conversation=ConversationConfig(
            num=num,
            turn=TurnConfig(mean=turn_mean),
        )
    )


# =============================================================================
# Serialization Tests
# =============================================================================


class TestUserConfigSerialization:
    """Tests for UserConfig serialization and deserialization."""

    def test_to_json_string(self):
        """Test serialization and deserialization to/from a JSON string."""
        config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["model1", "model2"],
                type=EndpointType.CHAT,
                custom_endpoint="custom_endpoint",
                streaming=True,
                urls=["http://custom-url"],
                api_key="test_api_key",
                timeout_seconds=10,
            ),
            input=InputConfig(
                random_seed=42,
                dataset_sampling_strategy=DatasetSamplingStrategy.SHUFFLE,
                extra=[("key1", "value1"), ("key2", "value2"), ("key3", "value3")],
                headers=[
                    ("Authorization", "Bearer token"),
                    ("Content-Type", "application/json"),
                ],
                conversation=ConversationConfig(
                    num=10,
                    turn=TurnConfig(
                        mean=10,
                        stddev=10,
                        delay=TurnDelayConfig(mean=10, stddev=10),
                    ),
                ),
            ),
            output=OutputConfig(artifact_directory="test_artifacts"),
            tokenizer=TokenizerConfig(name="test_tokenizer", revision="test_revision"),
            loadgen=LoadGeneratorConfig(concurrency=10, request_rate=10),
            cli_command="test_cli_command",
        )

        # Validate round-trip serialization
        assert (
            UserConfig.model_validate_json(
                config.model_dump_json(indent=4, exclude_unset=True)
            )
            == config
        )
        assert (
            UserConfig.model_validate_json(
                config.model_dump_json(indent=4, exclude_defaults=True)
            )
            == config
        )

    def test_to_file(self):
        """Test serialization and deserialization to/from a file."""
        config = make_config(
            endpoint=make_endpoint(streaming=True, url="http://custom-url"),
        )

        mocked_file = mock_open()
        with patch("pathlib.Path.open", mocked_file):
            mocked_file().write(config.model_dump_json(indent=4, exclude_defaults=True))

        with patch("pathlib.Path.open", mocked_file):
            mocked_file().read.return_value = config.model_dump_json(
                indent=4, exclude_defaults=True
            )
            loaded_config = UserConfig.model_validate_json(mocked_file().read())

        assert config == loaded_config

    def test_exclude_unset_fields(self):
        """Test that unset fields are correctly excluded when serializing."""
        config = make_config(
            endpoint=make_endpoint(streaming=True, url="http://custom-url"),
        )
        assert config.model_dump_json(exclude_unset=True) != config.model_dump_json()
        assert config.model_dump_json(exclude_defaults=True) != config.model_dump_json()
        assert (
            config.model_dump_json(exclude_unset=True, exclude_defaults=True)
            != config.model_dump_json()
        )
        assert config.model_dump_json(exclude_none=True) != config.model_dump_json()


class TestUserConfigDefaults:
    """Tests for UserConfig default values and custom values."""

    def test_defaults(self):
        """Test the default values of UserConfig."""
        config = make_config(endpoint=make_endpoint(model_names=["model1", "model2"]))

        assert config.endpoint.model_names == ["model1", "model2"]
        assert config.endpoint.streaming == EndpointDefaults.STREAMING
        assert config.endpoint.url == EndpointDefaults.URL
        assert isinstance(config.endpoint, EndpointConfig)
        assert isinstance(config.input, InputConfig)
        assert isinstance(config.output, OutputConfig)
        assert isinstance(config.tokenizer, TokenizerConfig)

    def test_custom_values(self):
        """Test UserConfig with custom values."""
        config = make_config(
            endpoint=make_endpoint(
                model_names=["model1", "model2"],
                streaming=True,
                url="http://custom-url",
            ),
        )

        assert config.endpoint.model_names == ["model1", "model2"]
        assert config.endpoint.streaming is True
        assert config.endpoint.url == "http://custom-url"
        assert isinstance(config.endpoint, EndpointConfig)
        assert isinstance(config.input, InputConfig)
        assert isinstance(config.output, OutputConfig)
        assert isinstance(config.tokenizer, TokenizerConfig)
        assert isinstance(config.loadgen, LoadGeneratorConfig)

    @pytest.mark.parametrize(
        "model_names,endpoint_type,timing_mode,streaming,expected_dir",
        [
            (["hf/model"], EndpointType.CHAT, TimingMode.REQUEST_RATE, True, "/tmp/artifacts/hf_model-openai-chat-concurrency5-request_rate10.0"),
            (["model1", "model2"], EndpointType.COMPLETIONS, TimingMode.REQUEST_RATE, True, "/tmp/artifacts/model1_multi-openai-completions-concurrency5-request_rate10.0"),
            (["singlemodel"], EndpointType.EMBEDDINGS, TimingMode.FIXED_SCHEDULE, False, "/tmp/artifacts/singlemodel-openai-embeddings-fixed_schedule"),
        ],
    )  # fmt: skip
    def test_compute_artifact_directory(
        self,
        monkeypatch,
        model_names,
        endpoint_type,
        timing_mode,
        streaming,
        expected_dir,
    ):
        """Test artifact directory computation based on config values."""
        endpoint = make_endpoint(
            endpoint_type=endpoint_type,
            model_names=model_names,
            streaming=streaming,
            url="http://custom-url",
        )
        output = OutputConfig(artifact_directory=Path("/tmp/artifacts"))
        loadgen = LoadGeneratorConfig(concurrency=5, request_rate=10, request_count=100)

        monkeypatch.setattr("pathlib.Path.is_file", lambda self: True)
        input_cfg = InputConfig(
            fixed_schedule=(timing_mode == TimingMode.FIXED_SCHEDULE),
            file="/tmp/dummy_input.txt",
        )
        config = UserConfig(
            endpoint=endpoint, output=output, loadgen=loadgen, input=input_cfg
        )
        monkeypatch.setattr(
            UserConfig, "_timing_mode", property(lambda self: timing_mode)
        )

        assert config._compute_artifact_directory() == Path(expected_dir)


# =============================================================================
# GPU Telemetry Configuration Tests
# =============================================================================


class TestGPUTelemetryConfig:
    """Tests for GPU telemetry configuration parsing and validation."""

    @pytest.mark.parametrize(
        "gpu_telemetry_input,expected_mode,expected_urls",
        [
            ([], GPUTelemetryMode.SUMMARY, []),
            (["dashboard"], GPUTelemetryMode.REALTIME_DASHBOARD, []),
            (["http://node1:9401/metrics"], GPUTelemetryMode.SUMMARY, ["http://node1:9401/metrics"]),
            (["dashboard", "http://node1:9401/metrics"], GPUTelemetryMode.REALTIME_DASHBOARD, ["http://node1:9401/metrics"]),
            (["http://node1:9401/metrics", "http://node2:9401/metrics"], GPUTelemetryMode.SUMMARY, ["http://node1:9401/metrics", "http://node2:9401/metrics"]),
            (["dashboard", "http://node1:9401/metrics", "http://node2:9401/metrics"], GPUTelemetryMode.REALTIME_DASHBOARD, ["http://node1:9401/metrics", "http://node2:9401/metrics"]),
            (["http://node1:9401/metrics", "dashboard"], GPUTelemetryMode.REALTIME_DASHBOARD, ["http://node1:9401/metrics"]),
        ],
    )  # fmt: skip
    def test_parse_config(self, gpu_telemetry_input, expected_mode, expected_urls):
        """Test parsing of gpu_telemetry list into mode and URLs."""
        config = make_config(gpu_telemetry=gpu_telemetry_input)

        assert config.gpu_telemetry_mode == expected_mode
        assert config.gpu_telemetry_urls == expected_urls

    def test_defaults(self):
        """Test that gpu_telemetry_mode and gpu_telemetry_urls have correct defaults."""
        config = make_config()

        assert config.gpu_telemetry_mode == GPUTelemetryMode.SUMMARY
        assert config.gpu_telemetry_urls == []

    def test_preserves_existing_fields(self):
        """Test that parsing GPU telemetry config doesn't affect other fields."""
        config = make_config(
            endpoint=make_endpoint(streaming=True),
            gpu_telemetry=["dashboard", "http://custom:9401/metrics"],
        )

        assert config.gpu_telemetry_mode == GPUTelemetryMode.REALTIME_DASHBOARD
        assert config.gpu_telemetry_urls == ["http://custom:9401/metrics"]
        assert config.endpoint.streaming is True
        assert config.endpoint.model_names == ["test-model"]

    def test_urls_extraction(self):
        """Test that only http URLs are extracted from gpu_telemetry list."""
        config = make_config(
            gpu_telemetry=[
                "dashboard",
                "http://node1:9401/metrics",
                "https://node2:9401/metrics",
                "summary",
            ],
        )

        assert len(config.gpu_telemetry_urls) == 2
        assert "http://node1:9401/metrics" in config.gpu_telemetry_urls
        assert "https://node2:9401/metrics" in config.gpu_telemetry_urls
        assert "dashboard" not in config.gpu_telemetry_urls
        assert "summary" not in config.gpu_telemetry_urls

    @pytest.mark.parametrize(
        "gpu_telemetry,expected_urls",
        [
            (
                [
                    "localhost:9400",
                    "node1:9401/metrics",
                    "http://node2:9400",
                    "https://node3:9401/metrics",
                ],
                [
                    "http://localhost:9400",
                    "http://node1:9401/metrics",
                    "http://node2:9400",
                    "https://node3:9401/metrics",
                ],
            ),
            (
                ["dashboard", "localhost:9400", "http://node1:9401"],
                ["http://localhost:9400", "http://node1:9401"],
            ),
        ],
    )
    def test_url_normalization(self, gpu_telemetry, expected_urls):
        """Test that URLs without http:// prefix are normalized correctly."""
        config = make_config(gpu_telemetry=gpu_telemetry)

        assert len(config.gpu_telemetry_urls) == len(expected_urls)
        for url in expected_urls:
            assert url in config.gpu_telemetry_urls

    def test_csv_file_not_found(self):
        """Test that GPU metrics CSV file validation raises error if file doesn't exist."""
        with pytest.raises(ValueError, match="GPU metrics file not found"):
            make_config(gpu_telemetry=["dashboard", "/nonexistent/path/metrics.csv"])


# =============================================================================
# Load Generator Validation Tests
# =============================================================================


class TestLoadGeneratorValidation:
    """Tests for LoadGeneratorConfig validation."""

    def test_arrival_pattern_conflict(self):
        """Test that CONCURRENCY_BURST mode with request_rate raises validation error."""
        with pytest.raises(
            ValueError,
            match="Request rate mode cannot be .* when a request rate is specified",
        ):
            make_config(
                loadgen=LoadGeneratorConfig(
                    request_rate=10.0,
                    arrival_pattern=ArrivalPattern.CONCURRENCY_BURST,
                ),
            )

    def test_benchmark_duration_and_count_together(self):
        """Test that both benchmark_duration and request_count can be set together."""
        config = make_config(
            loadgen=LoadGeneratorConfig(benchmark_duration=60, request_count=100),
        )
        assert config.loadgen.benchmark_duration == 60
        assert config.loadgen.request_count == 100

    def test_grace_period_without_duration(self):
        """Test that grace period without duration raises validation error."""
        with pytest.raises(
            ValueError,
            match="--benchmark-grace-period can only be used with duration-based benchmarking",
        ):
            make_config(loadgen=LoadGeneratorConfig(benchmark_grace_period=10))

    def test_multi_turn_request_count_conflict(self):
        """Test that both request_count and conversation num raises validation error."""
        with pytest.raises(
            ValueError,
            match="Both a request-count and number of conversations are set",
        ):
            make_config(
                input_config=InputConfig(conversation=ConversationConfig(num=50)),
                loadgen=LoadGeneratorConfig(request_count=100),
            )


# =============================================================================
# Concurrency Validation Tests
# =============================================================================


class TestConcurrencyValidation:
    """Tests for concurrency validation against request_count and conversation_num."""

    @pytest.mark.parametrize(
        "concurrency,request_count,should_raise",
        [
            (100, 50, True),   # exceeds
            (50, 50, False),   # equals
            (25, 100, False),  # less than
        ],
    )  # fmt: skip
    def test_vs_request_count_single_turn(
        self, concurrency, request_count, should_raise
    ):
        """Test concurrency validation against request_count for single-turn."""
        if should_raise:
            with pytest.raises(
                ValueError,
                match=f"Concurrency \\({concurrency}\\) cannot be greater than the request count \\({request_count}\\)",
            ):
                make_config(
                    loadgen=LoadGeneratorConfig(
                        concurrency=concurrency, request_count=request_count
                    ),
                )
        else:
            config = make_config(
                loadgen=LoadGeneratorConfig(
                    concurrency=concurrency, request_count=request_count
                ),
            )
            assert config.loadgen.concurrency == concurrency
            assert config.loadgen.request_count == request_count

    @pytest.mark.parametrize(
        "concurrency,conversation_num,should_raise",
        [
            (100, 50, True),   # exceeds
            (50, 50, False),   # equals
            (25, 100, False),  # less than
        ],
    )  # fmt: skip
    def test_vs_conversation_num_multi_turn(
        self, concurrency, conversation_num, should_raise
    ):
        """Test concurrency validation against conversation_num for multi-turn."""
        if should_raise:
            with pytest.raises(
                ValueError,
                match=f"Concurrency \\({concurrency}\\) cannot be greater than the number of conversations \\({conversation_num}\\)",
            ):
                make_config(
                    input_config=InputConfig(
                        conversation=ConversationConfig(num=conversation_num)
                    ),
                    loadgen=LoadGeneratorConfig(concurrency=concurrency),
                )
        else:
            config = make_config(
                input_config=InputConfig(
                    conversation=ConversationConfig(num=conversation_num)
                ),
                loadgen=LoadGeneratorConfig(concurrency=concurrency),
            )
            assert config.loadgen.concurrency == concurrency
            assert config.input.conversation.num == conversation_num

    def test_none_is_valid(self):
        """Test that concurrency=None doesn't trigger validation errors."""
        config = make_config(loadgen=LoadGeneratorConfig(request_count=50))
        assert config.loadgen.concurrency is None or config.loadgen.concurrency == 1

    def test_with_request_rate(self):
        """Test that concurrency validation works when request_rate is also specified."""
        with pytest.raises(
            ValueError,
            match="Concurrency \\(100\\) cannot be greater than the request count \\(50\\)",
        ):
            make_config(
                loadgen=LoadGeneratorConfig(
                    concurrency=100, request_count=50, request_rate=10.0
                ),
            )

    def test_with_default_request_count(self):
        """Test that concurrency validation applies even with default request_count."""
        with pytest.raises(
            ValueError, match="Concurrency.*cannot be greater than.*request count"
        ):
            make_config(loadgen=LoadGeneratorConfig(concurrency=100, request_rate=10.0))

    def test_with_duration_benchmarking(self):
        """Test that concurrency validation is skipped with duration-based benchmarking."""
        config = make_config(
            loadgen=LoadGeneratorConfig(concurrency=100, benchmark_duration=60),
        )
        assert config.loadgen.concurrency == 100
        assert config.loadgen.benchmark_duration == 60


# =============================================================================
# Rankings Configuration Tests
# =============================================================================


class TestRankingsConfig:
    """Tests for rankings endpoint configuration."""

    def test_passages_defaults_and_custom(self):
        """Test rankings passages mean and stddev defaults and custom values."""
        cfg_default = make_config(
            endpoint=make_endpoint(endpoint_type=EndpointType.HF_TEI_RANKINGS)
        )
        assert cfg_default.input.rankings.passages.mean == 1
        assert cfg_default.input.rankings.passages.stddev == 0

        cfg_custom = make_config(
            endpoint=make_endpoint(endpoint_type=EndpointType.HF_TEI_RANKINGS),
            input_config=InputConfig(
                rankings=RankingsConfig(
                    passages=RankingsPassagesConfig(mean=5, stddev=2)
                )
            ),
        )
        assert cfg_custom.input.rankings.passages.mean == 5
        assert cfg_custom.input.rankings.passages.stddev == 2

    @pytest.mark.parametrize("invalid_kwargs", [{"mean": 0}, {"stddev": -1}])
    def test_passages_validation_errors(self, invalid_kwargs):
        """Test that invalid rankings passages values raise validation errors."""
        with pytest.raises(ValidationError):
            make_config(
                endpoint=make_endpoint(endpoint_type=EndpointType.HF_TEI_RANKINGS),
                input_config=InputConfig(
                    rankings=RankingsConfig(
                        passages=RankingsPassagesConfig(**invalid_kwargs)
                    )
                ),
            )

    def test_passages_prompt_token_defaults_and_custom(self):
        """Test rankings passages prompt token defaults and custom values."""
        cfg_default = make_config(
            endpoint=make_endpoint(endpoint_type=EndpointType.HF_TEI_RANKINGS)
        )
        assert cfg_default.input.rankings.passages.prompt_token_mean == 550
        assert cfg_default.input.rankings.passages.prompt_token_stddev == 0

        cfg_custom = make_config(
            endpoint=make_endpoint(endpoint_type=EndpointType.HF_TEI_RANKINGS),
            input_config=InputConfig(
                rankings=RankingsConfig(
                    passages=RankingsPassagesConfig(
                        prompt_token_mean=100, prompt_token_stddev=10
                    )
                )
            ),
        )
        assert cfg_custom.input.rankings.passages.prompt_token_mean == 100
        assert cfg_custom.input.rankings.passages.prompt_token_stddev == 10

    def test_query_prompt_token_defaults_and_custom(self):
        """Test rankings query prompt token defaults and custom values."""
        cfg_default = make_config(
            endpoint=make_endpoint(endpoint_type=EndpointType.HF_TEI_RANKINGS)
        )
        assert cfg_default.input.rankings.query.prompt_token_mean == 550
        assert cfg_default.input.rankings.query.prompt_token_stddev == 0

        cfg_custom = make_config(
            endpoint=make_endpoint(endpoint_type=EndpointType.HF_TEI_RANKINGS),
            input_config=InputConfig(
                rankings=RankingsConfig(
                    query=RankingsQueryConfig(
                        prompt_token_mean=50, prompt_token_stddev=5
                    )
                )
            ),
        )
        assert cfg_custom.input.rankings.query.prompt_token_mean == 50
        assert cfg_custom.input.rankings.query.prompt_token_stddev == 5

    @pytest.mark.parametrize(
        "config_class,param_name,invalid_value",
        [
            (RankingsPassagesConfig, "prompt_token_mean", 0),
            (RankingsPassagesConfig, "prompt_token_stddev", -1),
            (RankingsQueryConfig, "prompt_token_mean", 0),
            (RankingsQueryConfig, "prompt_token_stddev", -1),
        ],
    )
    def test_prompt_token_validation_errors(
        self, config_class, param_name, invalid_value
    ):
        """Test that invalid rankings prompt token values raise validation errors."""
        with pytest.raises(ValidationError):
            if config_class == RankingsPassagesConfig:
                rankings_config = RankingsConfig(
                    passages=RankingsPassagesConfig(**{param_name: invalid_value})
                )
            else:
                rankings_config = RankingsConfig(
                    query=RankingsQueryConfig(**{param_name: invalid_value})
                )
            make_config(
                endpoint=make_endpoint(endpoint_type=EndpointType.HF_TEI_RANKINGS),
                input_config=InputConfig(rankings=rankings_config),
            )

    def test_and_prompt_tokens_cannot_be_set_together(self):
        """Test that prompt input tokens and rankings-specific token options cannot both be set."""
        with pytest.raises(ValidationError, match="cannot be used together"):
            make_config(
                endpoint=make_endpoint(endpoint_type=EndpointType.HF_TEI_RANKINGS),
                input_config=InputConfig(
                    prompt=PromptConfig(input_tokens=InputTokensConfig(mean=100)),
                    rankings=RankingsConfig(
                        passages=RankingsPassagesConfig(prompt_token_mean=200)
                    ),
                ),
            )

        with pytest.raises(ValidationError, match="cannot be used together"):
            make_config(
                endpoint=make_endpoint(endpoint_type=EndpointType.HF_TEI_RANKINGS),
                input_config=InputConfig(
                    prompt=PromptConfig(input_tokens=InputTokensConfig(stddev=10)),
                    rankings=RankingsConfig(
                        query=RankingsQueryConfig(prompt_token_mean=300)
                    ),
                ),
            )

    def test_tokens_only_is_allowed(self):
        """Test that setting only rankings-specific token options is allowed."""
        cfg = make_config(
            endpoint=make_endpoint(endpoint_type=EndpointType.HF_TEI_RANKINGS),
            input_config=InputConfig(
                rankings=RankingsConfig(
                    passages=RankingsPassagesConfig(
                        prompt_token_mean=100, prompt_token_stddev=10
                    ),
                    query=RankingsQueryConfig(
                        prompt_token_mean=50, prompt_token_stddev=5
                    ),
                )
            ),
        )
        assert cfg.input.rankings.passages.prompt_token_mean == 100
        assert cfg.input.rankings.passages.prompt_token_stddev == 10
        assert cfg.input.rankings.query.prompt_token_mean == 50
        assert cfg.input.rankings.query.prompt_token_stddev == 5

    def test_prompt_tokens_only_is_allowed(self):
        """Test that setting only prompt input tokens is allowed (no rankings options changed)."""
        cfg = make_config(
            input_config=InputConfig(
                prompt=PromptConfig(input_tokens=InputTokensConfig(mean=100))
            ),
        )
        assert cfg.input.prompt.input_tokens.mean == 100

    @pytest.mark.parametrize(
        "rankings_config",
        [
            RankingsConfig(passages=RankingsPassagesConfig(mean=5)),
            RankingsConfig(passages=RankingsPassagesConfig(stddev=2)),
            RankingsConfig(passages=RankingsPassagesConfig(prompt_token_mean=100)),
            RankingsConfig(passages=RankingsPassagesConfig(prompt_token_stddev=10)),
            RankingsConfig(query=RankingsQueryConfig(prompt_token_mean=50)),
            RankingsConfig(query=RankingsQueryConfig(prompt_token_stddev=5)),
        ],
    )
    def test_options_require_rankings_endpoint(self, rankings_config):
        """Test that rankings options cannot be used with non-rankings endpoints."""
        with pytest.raises(
            ValidationError, match="can only be used with rankings endpoint types"
        ):
            make_config(
                endpoint=make_endpoint(endpoint_type=EndpointType.CHAT),
                input_config=InputConfig(rankings=rankings_config),
            )

    @pytest.mark.parametrize(
        "endpoint_type",
        [EndpointType.COMPLETIONS, EndpointType.EMBEDDINGS, EndpointType.CHAT],
    )
    def test_options_rejected_for_non_rankings_endpoints(self, endpoint_type):
        """Test that rankings options are rejected for various non-rankings endpoint types."""
        with pytest.raises(
            ValidationError, match="can only be used with rankings endpoint types"
        ):
            make_config(
                endpoint=make_endpoint(endpoint_type=endpoint_type),
                input_config=InputConfig(
                    rankings=RankingsConfig(
                        passages=RankingsPassagesConfig(mean=5, prompt_token_mean=100)
                    )
                ),
            )

    @pytest.mark.parametrize(
        "endpoint_type",
        [
            EndpointType.COHERE_RANKINGS,
            EndpointType.HF_TEI_RANKINGS,
            EndpointType.NIM_RANKINGS,
        ],
    )
    def test_options_allowed_for_rankings_endpoints(self, endpoint_type):
        """Test that rankings options are allowed with rankings endpoint types."""
        cfg = make_config(
            endpoint=make_endpoint(endpoint_type=endpoint_type),
            input_config=InputConfig(
                rankings=RankingsConfig(
                    passages=RankingsPassagesConfig(
                        mean=5, stddev=2, prompt_token_mean=100, prompt_token_stddev=10
                    ),
                    query=RankingsQueryConfig(
                        prompt_token_mean=50, prompt_token_stddev=5
                    ),
                )
            ),
        )
        assert cfg.input.rankings.passages.mean == 5
        assert cfg.input.rankings.passages.stddev == 2
        assert cfg.input.rankings.passages.prompt_token_mean == 100
        assert cfg.input.rankings.passages.prompt_token_stddev == 10
        assert cfg.input.rankings.query.prompt_token_mean == 50
        assert cfg.input.rankings.query.prompt_token_stddev == 5


# ==============================================================================
# Context Prompt Validation Tests
# ==============================================================================


class TestContextPromptValidation:
    """Tests for context prompt validation (user_context vs shared_system)."""

    def test_user_context_requires_num_dataset_entries(self):
        """Test that user_context_prompt_length requires num_dataset_entries to be specified."""
        with pytest.raises(ValidationError) as exc_info:
            make_config(
                endpoint=make_endpoint(url="http://localhost:8000/v1/chat/completions"),
                input_config=InputConfig(
                    prompt={"prefix_prompt": {"user_context_prompt_length": 100}},
                ),
            )

        error = exc_info.value.errors()[0]
        assert "user-context-prompt-length" in error["msg"]
        assert "num-dataset-entries" in error["msg"]

    def test_user_context_with_num_dataset_entries_succeeds(self):
        """Test that user_context_prompt_length works when num_dataset_entries is specified."""
        config = make_config(
            endpoint=make_endpoint(url="http://localhost:8000/v1/chat/completions"),
            input_config=InputConfig(
                prompt={"prefix_prompt": {"user_context_prompt_length": 100}},
                conversation=ConversationConfig(num_dataset_entries=5),
            ),
        )

        assert config.input.prompt.prefix_prompt.user_context_prompt_length == 100
        assert config.input.conversation.num_dataset_entries == 5

    def test_shared_system_without_num_sessions_succeeds(self):
        """Test that shared_system_prompt_length works without num_sessions."""
        config = make_config(
            endpoint=make_endpoint(url="http://localhost:8000/v1/chat/completions"),
            input_config=InputConfig(
                prompt={"prefix_prompt": {"shared_system_prompt_length": 100}},
            ),
        )

        assert config.input.prompt.prefix_prompt.shared_system_prompt_length == 100

    @pytest.mark.parametrize(
        "prefix_prompt_config,error_keywords",
        [
            ({"shared_system_prompt_length": 100, "length": 50}, ["shared-system-prompt-length", "prefix-prompt-length"]),
            ({"user_context_prompt_length": 100, "length": 50}, ["user-context-prompt-length", "prefix-prompt-length"]),
            ({"shared_system_prompt_length": 100, "user_context_prompt_length": 50, "pool_size": 10}, ["mutually exclusive"]),
        ],
    )  # fmt: skip
    def test_mutually_exclusive_options(self, prefix_prompt_config, error_keywords):
        """Test that context prompts and legacy options are mutually exclusive."""
        with pytest.raises(ValidationError) as exc_info:
            input_kwargs: dict = {"prompt": {"prefix_prompt": prefix_prompt_config}}
            if "user_context_prompt_length" in prefix_prompt_config:
                input_kwargs["conversation"] = ConversationConfig(num_dataset_entries=5)

            make_config(
                endpoint=make_endpoint(url="http://localhost:8000/v1/chat/completions"),
                input_config=InputConfig(**input_kwargs),
            )

        error = exc_info.value.errors()[0]
        assert "mutually exclusive" in error["msg"]
        for keyword in error_keywords:
            if keyword != "mutually exclusive":
                assert keyword in error["msg"]

    def test_context_prompts_only_succeed(self):
        """Test that using only context prompts (no legacy options) works."""
        config = make_config(
            endpoint=make_endpoint(url="http://localhost:8000/v1/chat/completions"),
            input_config=InputConfig(
                prompt={
                    "prefix_prompt": {
                        "shared_system_prompt_length": 100,
                        "user_context_prompt_length": 50,
                    }
                },
                conversation=ConversationConfig(num_dataset_entries=5),
            ),
        )

        assert config.input.prompt.prefix_prompt.shared_system_prompt_length == 100
        assert config.input.prompt.prefix_prompt.user_context_prompt_length == 50
        assert config.input.prompt.prefix_prompt.length == 0
        assert config.input.prompt.prefix_prompt.pool_size == 0

    def test_legacy_prompts_only_succeed(self):
        """Test that using only legacy options (no context prompts) works."""
        config = make_config(
            endpoint=make_endpoint(url="http://localhost:8000/v1/chat/completions"),
            input_config=InputConfig(
                prompt={"prefix_prompt": {"length": 50, "pool_size": 10}}
            ),
        )

        assert config.input.prompt.prefix_prompt.length == 50
        assert config.input.prompt.prefix_prompt.pool_size == 10
        assert config.input.prompt.prefix_prompt.shared_system_prompt_length is None
        assert config.input.prompt.prefix_prompt.user_context_prompt_length is None


# =============================================================================
# Prefill Concurrency Validation Tests
# =============================================================================


class TestPrefillConcurrencyValidation:
    """Tests for prefill_concurrency validation."""

    @pytest.mark.parametrize(
        "streaming,prefill_concurrency,concurrency,should_raise,error_pattern",
        [
            (False, 5, 10, True, "--prefill-concurrency requires --streaming"),
            (True, 5, 10, False, None),
            (True, 10, 5, True, r"--prefill-concurrency \(10\) cannot be greater than --concurrency \(5\)"),
            (True, 10, 10, False, None),
            (True, 3, 10, False, None),
        ],
    )  # fmt: skip
    def test_validation(
        self, streaming, prefill_concurrency, concurrency, should_raise, error_pattern
    ):
        """Test prefill_concurrency requires streaming and cannot exceed concurrency."""
        if should_raise:
            with pytest.raises(ValidationError, match=error_pattern):
                make_config(
                    endpoint=make_endpoint(streaming=streaming),
                    loadgen=LoadGeneratorConfig(
                        concurrency=concurrency,
                        prefill_concurrency=prefill_concurrency,
                        request_count=100,
                    ),
                )
        else:
            config = make_config(
                endpoint=make_endpoint(streaming=streaming),
                loadgen=LoadGeneratorConfig(
                    concurrency=concurrency,
                    prefill_concurrency=prefill_concurrency,
                    request_count=100,
                ),
            )
            assert config.loadgen.prefill_concurrency == prefill_concurrency

    @pytest.mark.parametrize(
        "streaming,warmup_prefill,warmup_concurrency,concurrency,should_raise,error_pattern",
        [
            (False, 5, 10, 20, True, "--prefill-concurrency requires --streaming"),
            (True, 10, 5, 20, True, r"--warmup-prefill-concurrency \(10\) cannot be greater than warmup concurrency \(5\)"),
            (True, 15, None, 10, True, r"--warmup-prefill-concurrency \(15\) cannot be greater than warmup concurrency \(10\)"),
            (True, 8, 10, 5, False, None),
        ],
    )  # fmt: skip
    def test_warmup_validation(
        self,
        streaming,
        warmup_prefill,
        warmup_concurrency,
        concurrency,
        should_raise,
        error_pattern,
    ):
        """Test warmup_prefill_concurrency validation."""
        loadgen_kwargs = {
            "concurrency": concurrency,
            "warmup_prefill_concurrency": warmup_prefill,
            "warmup_request_count": 20,
            "request_count": 100,
        }
        if warmup_concurrency is not None:
            loadgen_kwargs["warmup_concurrency"] = warmup_concurrency

        if should_raise:
            with pytest.raises(ValidationError, match=error_pattern):
                make_config(
                    endpoint=make_endpoint(streaming=streaming),
                    loadgen=LoadGeneratorConfig(**loadgen_kwargs),
                )
        else:
            config = make_config(
                endpoint=make_endpoint(streaming=streaming),
                loadgen=LoadGeneratorConfig(**loadgen_kwargs),
            )
            assert config.loadgen.warmup_prefill_concurrency == warmup_prefill

    def test_none_is_valid(self):
        """Test that prefill_concurrency=None doesn't trigger validation errors."""
        config = make_config(
            endpoint=make_endpoint(streaming=False),
            loadgen=LoadGeneratorConfig(concurrency=10, request_count=100),
        )

        assert config.loadgen.prefill_concurrency is None


# =============================================================================
# Num Users Validation Tests
# =============================================================================


class TestNumUsersValidation:
    """Tests for num_users validation against request_count and num_sessions."""

    def test_num_users_none_no_validation(self):
        """Test that num_users=None doesn't trigger validation."""
        config = make_config(
            loadgen=LoadGeneratorConfig(request_count=50),
        )
        assert config.loadgen.num_users is None

    def test_num_users_without_request_count_or_sessions(self):
        """Test that num_users alone (no request_count or sessions) passes validation."""
        config = make_config(
            input_config=make_multi_turn_input(),
            loadgen=LoadGeneratorConfig(
                num_users=10, user_centric_rate=5.0, benchmark_duration=60
            ),
        )
        assert config.loadgen.num_users == 10

    @pytest.mark.parametrize(
        "num_users,request_count,should_raise",
        [
            (10, 5, True),    # num_users > request_count
            (10, 10, False),  # num_users == request_count
            (10, 20, False),  # num_users < request_count
        ],
    )  # fmt: skip
    def test_validation_vs_request_count(self, num_users, request_count, should_raise):
        """Test num_users validation against request_count."""
        if should_raise:
            with pytest.raises(
                ValueError,
                match=f"--request-count \\({request_count}\\) cannot be less than --num-users \\({num_users}\\)",
            ):
                make_config(
                    input_config=make_multi_turn_input(),
                    loadgen=LoadGeneratorConfig(
                        num_users=num_users,
                        user_centric_rate=5.0,
                        request_count=request_count,
                    ),
                )
        else:
            config = make_config(
                input_config=make_multi_turn_input(),
                loadgen=LoadGeneratorConfig(
                    num_users=num_users,
                    user_centric_rate=5.0,
                    request_count=request_count,
                ),
            )
            assert config.loadgen.num_users == num_users
            assert config.loadgen.request_count == request_count

    @pytest.mark.parametrize(
        "num_users,num_sessions,should_raise",
        [
            (10, 5, True),    # num_users > num_sessions
            (10, 10, False),  # num_users == num_sessions
            (10, 20, False),  # num_users < num_sessions
        ],
    )  # fmt: skip
    def test_validation_vs_num_sessions(self, num_users, num_sessions, should_raise):
        """Test num_users validation against num_sessions."""
        if should_raise:
            with pytest.raises(
                ValueError,
                match=f"--num-sessions \\({num_sessions}\\) cannot be less than --num-users \\({num_users}\\)",
            ):
                make_config(
                    input_config=make_multi_turn_input(num=num_sessions),
                    loadgen=LoadGeneratorConfig(
                        num_users=num_users, user_centric_rate=5.0
                    ),
                )
        else:
            config = make_config(
                input_config=make_multi_turn_input(num=num_sessions),
                loadgen=LoadGeneratorConfig(num_users=num_users, user_centric_rate=5.0),
            )
            assert config.loadgen.num_users == num_users
            assert config.input.conversation.num == num_sessions

    def test_validation_both_set_valid(self):
        """Test that when both request_count and num_sessions are individually valid.

        Note: We can't actually set both together due to validate_multi_turn_options,
        so we test them separately.
        """
        # Test with request_count
        config1 = make_config(
            input_config=make_multi_turn_input(),
            loadgen=LoadGeneratorConfig(
                num_users=10, user_centric_rate=5.0, request_count=30
            ),
        )
        assert config1.loadgen.num_users == 10
        assert config1.loadgen.request_count == 30

        # Test with num_sessions
        config2 = make_config(
            input_config=make_multi_turn_input(num=20),
            loadgen=LoadGeneratorConfig(num_users=10, user_centric_rate=5.0),
        )
        assert config2.loadgen.num_users == 10
        assert config2.input.conversation.num == 20

    def test_validation_both_set_sessions_invalid(self):
        """Test that num_sessions < num_users fails."""
        with pytest.raises(
            ValueError,
            match="--num-sessions \\(5\\) cannot be less than --num-users \\(10\\)",
        ):
            make_config(
                input_config=make_multi_turn_input(num=5),
                loadgen=LoadGeneratorConfig(num_users=10, user_centric_rate=5.0),
            )

    def test_validation_both_set_request_count_invalid(self):
        """Test that request_count < num_users fails."""
        with pytest.raises(
            ValueError,
            match="--request-count \\(5\\) cannot be less than --num-users \\(10\\)",
        ):
            make_config(
                input_config=make_multi_turn_input(),
                loadgen=LoadGeneratorConfig(
                    num_users=10, user_centric_rate=5.0, request_count=5
                ),
            )

    def test_validation_only_request_count_set(self):
        """Test validation when only request_count is set (no num_sessions)."""
        config = make_config(
            input_config=make_multi_turn_input(),
            loadgen=LoadGeneratorConfig(
                num_users=10, user_centric_rate=5.0, request_count=20
            ),
        )
        assert config.loadgen.num_users == 10
        assert config.loadgen.request_count == 20
        assert config.input.conversation.num is None

    def test_validation_only_num_sessions_set(self):
        """Test validation when only num_sessions is set (no request_count)."""
        config = make_config(
            input_config=make_multi_turn_input(num=15),
            loadgen=LoadGeneratorConfig(
                num_users=10, user_centric_rate=5.0, benchmark_duration=60
            ),
        )
        assert config.loadgen.num_users == 10
        assert config.input.conversation.num == 15
        # request_count may be set by other validators, so we just check it's not our concern


# =============================================================================
# User-Centric Rate Validation Tests
# =============================================================================


class TestUserCentricRateValidation:
    """Tests for user-centric rate mode validation.

    User-centric rate mode allows requests to "pile up" when server latency
    exceeds the inter-request interval. This is valid LMBenchmark behavior
    where multiple in-flight requests per user can occur simultaneously.
    """

    def test_user_centric_rate_requires_num_users(self):
        """Test that user_centric_rate requires num_users to be set."""
        with pytest.raises(
            ValueError, match="--user-centric-rate requires --num-users"
        ):
            make_config(
                loadgen=LoadGeneratorConfig(
                    user_centric_rate=10.0, benchmark_duration=60
                ),
            )

    def test_user_centric_rate_with_num_users_succeeds(self):
        """Test that user_centric_rate with num_users is valid."""
        config = make_config(
            input_config=make_multi_turn_input(),
            loadgen=LoadGeneratorConfig(
                num_users=10, user_centric_rate=5.0, benchmark_duration=60
            ),
        )
        assert config.loadgen.num_users == 10
        assert config.loadgen.user_centric_rate == 5.0
        assert config.timing_mode == TimingMode.USER_CENTRIC_RATE

    def test_user_centric_rate_cannot_use_request_rate(self):
        """Test that user_centric_rate cannot be combined with request_rate."""
        with pytest.raises(
            ValueError,
            match="--user-centric-rate cannot be used together with --request-rate",
        ):
            make_config(
                input_config=make_multi_turn_input(),
                loadgen=LoadGeneratorConfig(
                    num_users=10,
                    user_centric_rate=5.0,
                    request_rate=10.0,
                    benchmark_duration=60,
                ),
            )

    def test_user_centric_rate_cannot_use_arrival_pattern(self):
        """Test that user_centric_rate cannot be combined with arrival_pattern."""
        with pytest.raises(
            ValueError,
            match="--user-centric-rate cannot be used together with --request-rate or --arrival-pattern",
        ):
            make_config(
                input_config=make_multi_turn_input(),
                loadgen=LoadGeneratorConfig(
                    num_users=10,
                    user_centric_rate=5.0,
                    arrival_pattern=ArrivalPattern.CONSTANT,
                    benchmark_duration=60,
                ),
            )

    def test_user_centric_rate_with_concurrency_allowed(self):
        """Test that user_centric_rate can be used with concurrency.

        This allows "request pile-up" where multiple in-flight requests per user
        can occur when server latency exceeds the inter-request interval.
        """
        config = make_config(
            input_config=make_multi_turn_input(),
            loadgen=LoadGeneratorConfig(
                num_users=10,
                user_centric_rate=5.0,
                concurrency=20,  # Allow 2x pile-up
                benchmark_duration=60,
            ),
        )
        assert config.loadgen.num_users == 10
        assert config.loadgen.user_centric_rate == 5.0
        assert config.loadgen.concurrency == 20

    def test_user_centric_rate_with_prefill_concurrency_allowed(self):
        """Test that user_centric_rate can be used with prefill_concurrency.

        This supports scenarios where prefill stage concurrency needs to be
        limited while allowing request pile-up in decode stage.
        """
        config = make_config(
            input_config=make_multi_turn_input(),
            endpoint=make_endpoint(streaming=True),
            loadgen=LoadGeneratorConfig(
                num_users=10,
                user_centric_rate=5.0,
                concurrency=20,
                prefill_concurrency=5,  # Limit prefill stage
                benchmark_duration=60,
            ),
        )
        assert config.loadgen.num_users == 10
        assert config.loadgen.user_centric_rate == 5.0
        assert config.loadgen.concurrency == 20
        assert config.loadgen.prefill_concurrency == 5

    def test_user_centric_rate_high_pile_up_allowed(self):
        """Test that high pile-up (concurrency >> num_users) is allowed.

        In extreme latency scenarios, many requests per user may be in-flight.
        """
        config = make_config(
            input_config=make_multi_turn_input(),
            loadgen=LoadGeneratorConfig(
                num_users=5,
                user_centric_rate=100.0,  # High QPS
                concurrency=50,  # 10x pile-up per user
                benchmark_duration=60,
            ),
        )
        assert config.loadgen.num_users == 5
        assert config.loadgen.concurrency == 50
        # Pile-up ratio = 50 / 5 = 10x per user

    def test_user_centric_rate_with_num_sessions(self):
        """Test user_centric_rate with num_sessions stop condition."""
        config = make_config(
            input_config=make_multi_turn_input(num=20),
            loadgen=LoadGeneratorConfig(
                num_users=10,
                user_centric_rate=5.0,
            ),
        )
        assert config.loadgen.num_users == 10
        assert config.loadgen.user_centric_rate == 5.0
        assert config.input.conversation.num == 20

    def test_user_centric_rate_with_request_count(self):
        """Test user_centric_rate with request_count stop condition."""
        config = make_config(
            input_config=make_multi_turn_input(),
            loadgen=LoadGeneratorConfig(
                num_users=10,
                user_centric_rate=5.0,
                request_count=100,
            ),
        )
        assert config.loadgen.num_users == 10
        assert config.loadgen.user_centric_rate == 5.0
        assert config.loadgen.request_count == 100
