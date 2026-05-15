# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.common.config import (
    ConversationConfig,
    EndpointConfig,
    EndpointDefaults,
    InputConfig,
    LoadGeneratorConfig,
    OutputConfig,
    PrefixPromptConfig,
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
    GPUTelemetryCollectorType,
    TimingMode,
)
from tests.harness import mock_plugin

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

        # Validate round-trip serialization (include_secrets to preserve api_key)
        ctx = {"include_secrets": True}
        assert (
            UserConfig.model_validate_json(
                config.model_dump_json(indent=4, exclude_unset=True, context=ctx)
            )
            == config
        )
        assert (
            UserConfig.model_validate_json(
                config.model_dump_json(indent=4, exclude_defaults=True, context=ctx)
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

    def test_compute_artifact_directory_with_concurrency_list(self, monkeypatch):
        """Test artifact directory uses concurrency_sweep_ prefix for list concurrency.

        Regression test: list concurrency previously rendered as
        'concurrency[2, 4, 8]' with shell-hostile brackets and spaces.
        """
        endpoint = make_endpoint(
            endpoint_type=EndpointType.CHAT,
            model_names=["test-model"],
            streaming=True,
            url="http://custom-url",
        )
        output = OutputConfig(artifact_directory=Path("/tmp/artifacts"))
        loadgen = LoadGeneratorConfig(concurrency=[2, 4, 8], request_count=100)

        config = UserConfig(endpoint=endpoint, output=output, loadgen=loadgen)

        artifact_dir = str(config._compute_artifact_directory())
        assert "concurrency_sweep_2_4_8" in artifact_dir
        assert "[" not in artifact_dir
        assert "]" not in artifact_dir


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

    def test_amdsmi_collector_selected_when_amdsmi_keyword_provided(self) -> None:
        """`--gpu-telemetry amdsmi` selects the AMDSMI collector type."""
        # Inject a fake amdsmi module so the import_module probe succeeds
        # without requiring ROCm installed on the test host.
        sys.modules["amdsmi"] = type(sys)("amdsmi")
        try:
            config = make_config(gpu_telemetry=["amdsmi"])
            assert (
                config._gpu_telemetry_collector_type == GPUTelemetryCollectorType.AMDSMI
            )
        finally:
            sys.modules.pop("amdsmi", None)

    def test_conflicting_local_collector_keywords_raise(self) -> None:
        """Passing both `pynvml` and `amdsmi` raises rather than last-wins."""
        sys.modules["amdsmi"] = type(sys)("amdsmi")
        sys.modules["pynvml"] = type(sys)("pynvml")
        try:
            with pytest.raises(
                ValidationError, match="Conflicting local GPU telemetry collectors"
            ):
                make_config(gpu_telemetry=["pynvml", "amdsmi"])
        finally:
            sys.modules.pop("amdsmi", None)
            sys.modules.pop("pynvml", None)

    def test_amdsmi_with_dcgm_url_raises(self) -> None:
        """Combining --gpu-telemetry amdsmi with a DCGM URL raises a clear error."""
        sys.modules["amdsmi"] = type(sys)("amdsmi")
        try:
            with pytest.raises(
                ValidationError, match="Cannot use amdsmi with DCGM URLs"
            ):
                make_config(gpu_telemetry=["amdsmi", "http://localhost:9400"])
        finally:
            sys.modules.pop("amdsmi", None)

    def test_amdsmi_missing_module_yields_install_hint(self) -> None:
        """validate_environment surfaces the install hint when amdsmi is absent."""
        from aiperf.gpu_telemetry import amdsmi_collector as amdsmi_mod

        with (
            patch.object(amdsmi_mod, "amdsmi", None),
            pytest.raises(ValidationError, match="amdsmi package not installed"),
        ):
            make_config(gpu_telemetry=["amdsmi"])

    def test_amdsmi_native_lib_oserror_yields_install_hint(self) -> None:
        """validate_environment surfaces the install hint when amdsmi native lib fails to load."""
        from aiperf.gpu_telemetry import amdsmi_collector as amdsmi_mod

        # The module-level try/except already coerces ImportError/OSError into
        # ``amdsmi is None``; validate_environment then raises the hint.
        with (
            patch.object(amdsmi_mod, "amdsmi", None),
            pytest.raises(ValidationError, match="amdsmi package not installed"),
        ):
            make_config(gpu_telemetry=["amdsmi"])

    def test_invalid_gpu_telemetry_item_message_has_no_leading_comma_without_local_collectors(
        self,
    ) -> None:
        with (
            patch(
                "aiperf.common.config.user_config._local_collector_keywords",
                return_value={},
            ),
            pytest.raises(ValidationError) as exc_info,
        ):
            make_config(gpu_telemetry=["not_a_real_keyword"])

        assert "Valid options are: 'dashboard', '.csv' file, and URLs." in str(
            exc_info.value
        )
        assert "Valid options are: ," not in str(exc_info.value)

    def test_selection_with_runtime_local_collector(self) -> None:
        fake_name = "fake_local_gpu"
        fake_enum_member = "FAKE_LOCAL_GPU"
        validated_environment = False

        class FakeLocalCollector:
            @classmethod
            def validate_environment(cls) -> None:
                nonlocal validated_environment
                validated_environment = True

        GPUTelemetryCollectorType.register(fake_enum_member, fake_name)
        try:
            with mock_plugin(
                "gpu_telemetry_collector",
                fake_name,
                FakeLocalCollector,
                metadata={"is_local": True},
            ):
                config = make_config(gpu_telemetry=[fake_name])
                assert config.gpu_telemetry_collector_type == GPUTelemetryCollectorType(
                    fake_name
                )
                assert validated_environment is True
        finally:
            GPUTelemetryCollectorType.deregister(fake_enum_member)

    def test_runtime_local_collector_validation_error_surfaces(self) -> None:
        fake_name = "fake_broken_gpu"
        fake_enum_member = "FAKE_BROKEN_GPU"

        class FakeBrokenCollector:
            @classmethod
            def validate_environment(cls) -> None:
                raise RuntimeError("fake_broken_gpu bindings are not installed")

        GPUTelemetryCollectorType.register(fake_enum_member, fake_name)
        try:
            with (
                mock_plugin(
                    "gpu_telemetry_collector",
                    fake_name,
                    FakeBrokenCollector,
                    metadata={"is_local": True},
                ),
                pytest.raises(
                    ValidationError, match="fake_broken_gpu bindings are not installed"
                ),
            ):
                make_config(gpu_telemetry=[fake_name])
        finally:
            GPUTelemetryCollectorType.deregister(fake_enum_member)

    def test_conflict_detection_with_runtime_local_collector(self) -> None:
        fake_name = "fake_local_gpu"
        fake_enum_member = "FAKE_LOCAL_GPU"

        class FakeLocalCollector:
            @classmethod
            def validate_environment(cls) -> None:
                """No-op; fake collector has no native bindings to probe."""

        GPUTelemetryCollectorType.register(fake_enum_member, fake_name)
        try:
            with mock_plugin(
                "gpu_telemetry_collector",
                fake_name,
                FakeLocalCollector,
                metadata={"is_local": True},
            ):
                sys.modules["pynvml"] = type(sys)("pynvml")
                try:
                    with pytest.raises(
                        ValidationError,
                        match="Conflicting local GPU telemetry collectors",
                    ):
                        make_config(gpu_telemetry=["pynvml", fake_name])
                finally:
                    sys.modules.pop("pynvml", None)
        finally:
            GPUTelemetryCollectorType.deregister(fake_enum_member)

    def test_local_vs_url_guardrail_for_runtime_local_collector(self) -> None:
        fake_name = "fake_local_gpu"
        fake_enum_member = "FAKE_LOCAL_GPU"

        class FakeLocalCollector:
            @classmethod
            def validate_environment(cls) -> None:
                """No-op; fake collector has no native bindings to probe."""

        GPUTelemetryCollectorType.register(fake_enum_member, fake_name)
        try:
            with (
                mock_plugin(
                    "gpu_telemetry_collector",
                    fake_name,
                    FakeLocalCollector,
                    metadata={"is_local": True},
                ),
                pytest.raises(
                    ValidationError,
                    match=f"Cannot use {fake_name} with DCGM URLs",
                ),
            ):
                make_config(gpu_telemetry=[fake_name, "http://localhost:9400"])
        finally:
            GPUTelemetryCollectorType.deregister(fake_enum_member)

    def test_invalid_item_message_includes_runtime_keyword(self) -> None:
        fake_name = "fake_local_gpu"
        fake_enum_member = "FAKE_LOCAL_GPU"

        class FakeLocalCollector:
            @classmethod
            def validate_environment(cls) -> None:
                """No-op; fake collector has no native bindings to probe."""

        GPUTelemetryCollectorType.register(fake_enum_member, fake_name)
        try:
            with mock_plugin(
                "gpu_telemetry_collector",
                fake_name,
                FakeLocalCollector,
                metadata={"is_local": True},
            ):
                with pytest.raises(
                    ValidationError, match=f"Invalid GPU telemetry item.*'{fake_name}'"
                ) as exc_info:
                    make_config(gpu_telemetry=["not_a_real_keyword"])
                assert f"'{fake_name}'" in str(exc_info.value)
        finally:
            GPUTelemetryCollectorType.deregister(fake_enum_member)

    def test_urls_extraction(self):
        """Test that only http URLs are extracted from gpu_telemetry list."""
        config = make_config(
            gpu_telemetry=[
                "dashboard",
                "http://node1:9401/metrics",
                "https://node2:9401/metrics",
            ],
        )

        assert len(config.gpu_telemetry_urls) == 2
        assert "http://node1:9401/metrics" in config.gpu_telemetry_urls
        assert "https://node2:9401/metrics" in config.gpu_telemetry_urls
        assert "dashboard" not in config.gpu_telemetry_urls

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

    def test_pynvml_with_urls_raises_error(self):
        """Test that using pynvml with DCGM URLs raises an error."""
        with pytest.raises(ValueError, match="Cannot use pynvml with DCGM URLs"):
            make_config(gpu_telemetry=["pynvml", "http://localhost:9401/metrics"])

    def test_pynvml_with_multiple_urls_raises_error(self):
        """Test that using pynvml with multiple DCGM URLs raises an error."""
        with pytest.raises(ValueError, match="Cannot use pynvml with DCGM URLs"):
            make_config(
                gpu_telemetry=[
                    "pynvml",
                    "http://node1:9401/metrics",
                    "http://node2:9401/metrics",
                ]
            )

    def test_pynvml_with_dashboard_allowed(self):
        """Test that pynvml can be used with dashboard mode."""
        config = make_config(gpu_telemetry=["pynvml", "dashboard"])

        assert config.gpu_telemetry_collector_type == GPUTelemetryCollectorType.PYNVML
        assert config.gpu_telemetry_mode == GPUTelemetryMode.REALTIME_DASHBOARD
        assert config.gpu_telemetry_urls == []

    def test_pynvml_only(self):
        """Test that pynvml can be used alone."""
        config = make_config(gpu_telemetry=["pynvml"])

        assert config.gpu_telemetry_collector_type == GPUTelemetryCollectorType.PYNVML
        assert config.gpu_telemetry_mode == GPUTelemetryMode.SUMMARY
        assert config.gpu_telemetry_urls == []

    @pytest.mark.parametrize(
        "url,should_warn",
        [
            ("localhost:8000", False),
            ("http://localhost:8000", False),
            ("127.0.0.1:8000", False),
            ("http://127.0.0.1:8000", False),
            ("::1:8000", False),
            ("http://[::1]:8000", False),
            ("remote-server:8000", True),
            ("http://remote-server:8000", True),
            ("192.168.1.100:8000", True),
            ("http://192.168.1.100:8000", True),
        ],
    )
    def test_pynvml_warns_on_non_localhost_url(self, url, should_warn, caplog):
        """Test that pynvml with non-localhost server URLs emits a warning."""
        import logging

        caplog.set_level(logging.WARNING)

        make_config(
            endpoint=make_endpoint(url=url),
            gpu_telemetry=["pynvml"],
        )

        warning_present = any(
            "pynvml collects GPU metrics from the local machine only" in record.message
            for record in caplog.records
        )
        assert warning_present == should_warn, (
            f"Expected warning={'present' if should_warn else 'absent'} for URL {url}, "
            f"but got {'present' if warning_present else 'absent'}"
        )

    def test_pynvml_warns_lists_non_localhost_urls(self, caplog):
        """Test that the warning lists the non-localhost URLs."""
        import logging

        caplog.set_level(logging.WARNING)

        make_config(
            endpoint=make_endpoint(
                urls=["http://remote-server:8000", "http://other-server:8000"]
            ),
            gpu_telemetry=["pynvml"],
        )

        warning_messages = [r.message for r in caplog.records if "pynvml" in r.message]
        assert len(warning_messages) == 1
        assert "remote-server" in warning_messages[0]
        assert "other-server" in warning_messages[0]

    def test_pynvml_no_warn_on_localhost_only(self, caplog):
        """Test that pynvml with only localhost URLs does not emit a warning."""
        import logging

        caplog.set_level(logging.WARNING)

        make_config(
            endpoint=make_endpoint(
                urls=["http://localhost:8000", "http://127.0.0.1:8001"]
            ),
            gpu_telemetry=["pynvml"],
        )

        warning_present = any(
            "pynvml collects GPU metrics from the local machine only" in record.message
            for record in caplog.records
        )
        assert not warning_present

    @pytest.mark.parametrize(
        "invalid_item",
        [
            "unknown",
            "invalid_option",
            "dcgm",
            "gpu",
            "telemetry",
            "metrics",
        ],
    )
    def test_unknown_item_raises_error(self, invalid_item):
        """Test that unknown items in gpu_telemetry raise an error."""
        with pytest.raises(ValueError, match="Invalid GPU telemetry item"):
            make_config(gpu_telemetry=[invalid_item])


class TestIsLocalhostUrl:
    """Direct tests for the `_is_localhost_url` private helper.

    `EndpointConfig` now prepends `http://` to scheme-less URLs (fix for the
    readiness-probe bug). This helper has to recognize both the pre- and
    post-normalization forms — including the IPv6-without-brackets edge case
    that pre-existed the fix and would otherwise have regressed.
    """

    @pytest.mark.parametrize(
        "url,expected",
        [
            # localhost / 127.0.0.1 — common forms
            pytest.param("localhost", True, id="bare_localhost"),
            pytest.param("localhost:8000", True, id="localhost_port"),
            pytest.param("http://localhost", True, id="localhost_http"),
            pytest.param("http://localhost:8000", True, id="localhost_http_port"),
            pytest.param("https://localhost:8443", True, id="localhost_https_port"),
            pytest.param("127.0.0.1", True, id="bare_127"),
            pytest.param("127.0.0.1:8000", True, id="127_port"),
            pytest.param("http://127.0.0.1:8000", True, id="127_http_port"),
            # IPv6 ::1 with brackets — well-formed
            pytest.param("[::1]:8000", True, id="ipv6_bracketed_port"),
            pytest.param("http://[::1]:8000", True, id="ipv6_bracketed_http"),
            pytest.param("https://[::1]:8443", True, id="ipv6_bracketed_https"),
            # IPv6 ::1 without brackets — pre- and post-normalization forms
            pytest.param("::1:8000", True, id="ipv6_bare_pre_normalization"),
            pytest.param("http://::1:8000", True, id="ipv6_bare_post_normalization"),
            pytest.param(
                "https://::1:8443", True, id="ipv6_bare_https_post_normalization"
            ),
            # External hosts — must NOT match localhost
            pytest.param("remote-server:8000", False, id="remote_host"),
            pytest.param("http://remote-server:8000", False, id="remote_host_http"),
            pytest.param("192.168.1.100:8000", False, id="lan_ipv4"),
            pytest.param("http://192.168.1.100:8000", False, id="lan_ipv4_http"),
            pytest.param("http://example.com", False, id="public_dns"),
        ],
    )
    def test_is_localhost_url(self, url: str, expected: bool) -> None:
        from aiperf.common.config.user_config import _is_localhost_url

        assert _is_localhost_url(url) is expected

    def test_unknown_item_with_valid_items_raises_error(self):
        """Test that unknown items mixed with valid items still raise an error."""
        with pytest.raises(ValueError, match="Invalid GPU telemetry item"):
            make_config(gpu_telemetry=["dashboard", "unknown_option"])


# =============================================================================
# OTel Streaming Configuration Tests
# =============================================================================


class TestOTelStreamingConfig:
    """Tests for OTel streaming configuration parsing and validation."""

    def test_disabled_by_default(self):
        """OTel streaming is disabled when no collector URL is provided."""
        config = make_config()
        assert config.otel_collector_enabled is False
        assert config.otel_metrics_url is None
        assert config.otel_stream_metrics_enabled is True
        assert config.otel_stream_timing_enabled is True

    @pytest.mark.parametrize(
        "otel_url,expected_url",
        [
            ("collector:4318", "http://collector:4318/v1/metrics"),
            ("http://collector:4318", "http://collector:4318/v1/metrics"),
            (
                "http://collector:4318/v1/metrics",
                "http://collector:4318/v1/metrics",
            ),
            ("https://collector:4318/otlp", "https://collector:4318/otlp/v1/metrics"),
        ],
    )
    def test_url_normalization(self, otel_url: str, expected_url: str):
        """Collector URL is normalized to an OTLP metrics endpoint."""
        config = make_config(otel_url=otel_url)
        assert config.otel_collector_enabled is True
        assert config.otel_metrics_url == expected_url

    @pytest.mark.parametrize("invalid_otel_url", ["", "   "])
    def test_invalid_empty_url(self, invalid_otel_url: str):
        """Empty collector URL is rejected."""
        with pytest.raises(ValueError, match="--otel-url cannot be empty"):
            make_config(otel_url=invalid_otel_url)

    @pytest.mark.parametrize(
        "invalid_otel_url",
        ["ftp://collector:4318", "file://collector:4318"],
    )
    def test_invalid_otel_scheme(self, invalid_otel_url: str):
        """Unsupported collector URL schemes are rejected."""
        with pytest.raises(ValueError, match="Invalid --otel-url value"):
            make_config(otel_url=invalid_otel_url)

    def test_grpc_scheme_error_names_http_https(self):
        """Rejecting grpc:// must tell the user http/https are the supported schemes.

        Regression for the manual-verify checklist item B4 where users reach for
        grpc://collector:4317 (the OTLP/gRPC default port) and need to know AIPerf
        ships only the OTLP/HTTP exporter.
        """
        with pytest.raises(
            ValueError,
            match=r"[Oo]nly http and https schemes are supported.*'grpc'",
        ):
            make_config(otel_url="grpc://collector:4317")

    @pytest.mark.parametrize(
        "bare_port_url",
        [
            param("http://:4318", id="scheme-colon-port"),
            param("https://:4318", id="https-scheme-colon-port"),
        ],
    )  # fmt: skip
    def test_empty_hostname_is_rejected(self, bare_port_url: str):
        """urlparse('http://:4318') has truthy netloc=':4318' but hostname=None.
        The validator must require a non-empty hostname so bare-port URLs don't
        slip through and create a malformed OTLP endpoint.
        """
        with pytest.raises(ValueError, match="Invalid --otel-url value"):
            make_config(otel_url=bare_port_url)

    @pytest.mark.parametrize(
        "stream_value,metrics_enabled,timing_enabled",
        [
            ("metrics", True, False),
            ("timing", False, True),
            ("default", True, True),
            (["metrics", "timing"], True, True),
            (["metrics", "timing", "default"], True, True),
        ],
    )
    def test_otel_stream_selection(
        self,
        stream_value: str | list[str],
        metrics_enabled: bool,
        timing_enabled: bool,
    ):
        """OTel stream selection parses into metrics/timing flags."""
        config = make_config(
            otel_url="collector:4318",
            stream=stream_value,
        )
        assert config.otel_stream_metrics_enabled is metrics_enabled
        assert config.otel_stream_timing_enabled is timing_enabled

    @pytest.mark.parametrize(
        "stream_value,error_pattern",
        [
            ("unknown", "Invalid --stream value"),
            ("both", "Valid options are: metrics, timing, default"),
            ("none", "Valid options are: metrics, timing, default"),
            (["metrics", "bad"], "Invalid --stream value"),
        ],
    )
    def test_otel_stream_invalid_values(
        self, stream_value: str | list[str], error_pattern: str
    ):
        """Invalid OTel stream selections are rejected."""
        with pytest.raises(ValueError, match=error_pattern):
            make_config(
                otel_url="collector:4318",
                stream=stream_value,
            )

    def test_otel_stream_defaults_to_both_when_unset_with_otel_url(self):
        """Omitting --stream keeps both domains enabled."""
        config = make_config(otel_url="collector:4318")
        assert config.otel_stream_metrics_enabled is True
        assert config.otel_stream_timing_enabled is True

    @pytest.mark.parametrize(
        "resource_attrs,bad_token",
        [
            param(["team"], "team", id="missing-equals"),
            param(["=prod"], "=prod", id="empty-key"),
            param(["team="], "team=", id="empty-value"),
            param(["env=prod,team"], "team", id="missing-equals-in-comma-list"),
            param(["env=prod,=us-west"], "=us-west", id="empty-key-in-comma-list"),
            param(["env=prod,team="], "team=", id="empty-value-in-comma-list"),
        ],
    )  # fmt: skip
    def test_otel_resource_attributes_rejects_malformed_entries(
        self, resource_attrs: list[str], bad_token: str
    ):
        """Regression: --otel-resource-attributes is documented as key=value,
        but malformed entries were previously accepted silently — missing ``=``
        was dropped, ``=prod`` emitted an empty key, and ``team=`` emitted an
        empty value. Each must now raise a ValueError pinpointing the bad token.
        """
        with pytest.raises(ValueError, match=r"--otel-resource-attributes"):
            make_config(
                otel_url="collector:4318",
                otel_resource_attributes=resource_attrs,
            )

    def test_otel_resource_attributes_accepts_well_formed_entries(self):
        """Well-formed comma-separated and repeated entries still parse."""
        config = make_config(
            otel_url="collector:4318",
            otel_resource_attributes=["team=inference", "env=prod,region=us-west-2"],
        )
        assert config.otel_custom_resource_attributes == {
            "team": "inference",
            "env": "prod",
            "region": "us-west-2",
        }

    @pytest.mark.parametrize(
        "secondary_kwargs",
        [
            param({"stream": "metrics"}, id="stream"),
            param({"otel_resource_attributes": ["env=prod"]}, id="resource-attrs"),
            param({"gen_ai_provider": "my-provider"}, id="gen-ai-provider"),
        ],
    )  # fmt: skip
    def test_otel_secondary_flags_require_otel_url(
        self, secondary_kwargs: dict[str, object]
    ):
        """--stream, --otel-resource-attributes, and --gen-ai-provider are
        consumed only by the OTel pipeline. Setting any of them without
        --otel-url is a silent no-op, so the validator rejects it.
        Regression for --gen-ai-provider being silently ignored.
        """
        with pytest.raises(
            ValueError,
            match=r"--stream, --otel-resource-attributes, and --gen-ai-provider",
        ):
            make_config(**secondary_kwargs)


class TestMLflowConfig:
    """Tests for MLflow post-run export configuration parsing and validation."""

    def test_disabled_by_default(self):
        config = make_config()
        assert config.mlflow_enabled is False
        assert config.mlflow_tracking_uri is None

    @pytest.mark.parametrize(
        "tracking_uri,experiment,run_name",
        [
            ("http://localhost:5000", "aiperf", None),
            (" https://mlflow.internal ", " perf-runs ", " nightly "),
        ],
    )
    def test_mlflow_normalization(
        self, tracking_uri: str, experiment: str, run_name: str | None
    ):
        config = make_config(
            mlflow_tracking_uri=tracking_uri,
            mlflow_experiment=experiment,
            mlflow_run_name=run_name,
        )
        assert config.mlflow_enabled is True
        assert config.mlflow_tracking_uri == tracking_uri.strip()
        assert config.mlflow_experiment == experiment.strip()
        if run_name is None:
            assert config.mlflow_run_name is None
        else:
            assert config.mlflow_run_name == run_name.strip()

    @pytest.mark.parametrize("invalid_uri", ["", "   "])
    def test_mlflow_tracking_uri_cannot_be_empty(self, invalid_uri: str):
        with pytest.raises(ValueError, match="--mlflow-tracking-uri cannot be empty"):
            make_config(mlflow_tracking_uri=invalid_uri)

    def test_mlflow_experiment_cannot_be_empty_when_enabled(self):
        with pytest.raises(
            ValueError,
            match="--mlflow-experiment cannot be empty when --mlflow-tracking-uri is set",
        ):
            make_config(
                mlflow_tracking_uri="http://localhost:5000",
                mlflow_experiment="   ",
            )

    def test_mlflow_tags_parse_to_dict(self):
        config = make_config(
            mlflow_tracking_uri="http://localhost:5000",
            mlflow_tags="team:perf,env:ci",
        )
        assert config.mlflow_tags_dict == {"team": "perf", "env": "ci"}

    def test_mlflow_artifact_glob_defaults(self):
        config = make_config(mlflow_tracking_uri="http://localhost:5000")
        assert config.mlflow_resolved_artifact_globs
        assert "**/*.png" in config.mlflow_resolved_artifact_globs

    def test_mlflow_artifact_glob_override(self):
        config = make_config(
            mlflow_tracking_uri="http://localhost:5000",
            mlflow_artifact_globs=["reports/*.json", "plots/*.png"],
        )
        assert config.mlflow_resolved_artifact_globs == [
            "reports/*.json",
            "plots/*.png",
        ]

    def test_mlflow_artifact_glob_entries_are_stripped(self):
        config = make_config(
            mlflow_tracking_uri="http://localhost:5000",
            mlflow_artifact_globs=[" reports/*.json ", " plots/*.png "],
        )
        assert config.mlflow_resolved_artifact_globs == [
            "reports/*.json",
            "plots/*.png",
        ]

    def test_mlflow_artifact_glob_empty_entry_is_rejected(self):
        with pytest.raises(
            ValueError, match="--mlflow-artifact-glob entries cannot be empty"
        ):
            make_config(
                mlflow_tracking_uri="http://localhost:5000",
                mlflow_artifact_globs=["reports/*.json", "   "],
            )

    def test_mlflow_tracking_uri_without_enable_flag_auto_enables(self):
        config = make_config(mlflow_tracking_uri="http://localhost:5000")
        assert config.mlflow_enabled is True
        assert config.mlflow_tracking_uri == "http://localhost:5000"

    @pytest.mark.parametrize(
        "secondary_kwargs",
        [
            param({"mlflow_experiment": "pre-seeded-experiment"}, id="experiment"),
            param({"mlflow_run_name": "my-run"}, id="run-name"),
            param({"mlflow_tags": "team:perf"}, id="tags"),
            param({"mlflow_artifact_globs": ["plots/*.png"]}, id="artifact-globs"),
            param({"mlflow_parent_run_id": "parent-123"}, id="parent-run-id"),
        ],
    )  # fmt: skip
    def test_mlflow_secondary_flags_require_tracking_uri(
        self, secondary_kwargs: dict[str, object]
    ):
        """Secondary MLflow flags without --mlflow-tracking-uri are a silent no-op,
        so the validator rejects them — mirrors the OTel secondary-flag behavior.
        """
        with pytest.raises(
            ValueError,
            match=r"require --mlflow-tracking-uri to be set",
        ):
            make_config(**secondary_kwargs)


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


class TestNonTokenizingEndpointValidation:
    """Tests for default_no_text_for_non_tokenizing_endpoints validator."""

    @staticmethod
    def _make_image_retrieval(**kwargs) -> UserConfig:
        return make_config(
            endpoint=make_endpoint(endpoint_type=EndpointType.IMAGE_RETRIEVAL),
            **kwargs,
        )

    def test_defaults_input_tokens_mean_to_zero(self):
        config = self._make_image_retrieval()
        assert config.input.prompt.input_tokens.mean == 0

    def test_defaults_input_tokens_stddev_to_zero(self):
        config = self._make_image_retrieval()
        assert config.input.prompt.input_tokens.stddev == 0

    def test_defaults_batch_size_to_zero(self):
        config = self._make_image_retrieval()
        assert config.input.prompt.batch_size == 0

    def test_rejects_explicit_input_tokens_stddev(self):
        with pytest.raises(ValidationError, match="--synthetic-input-tokens-stddev"):
            self._make_image_retrieval(
                input_config=InputConfig(
                    prompt=PromptConfig(input_tokens=InputTokensConfig(stddev=32))
                ),
            )

    def test_rejects_explicit_input_tokens_mean(self):
        with pytest.raises(ValidationError, match="--synthetic-input-tokens-mean"):
            self._make_image_retrieval(
                input_config=InputConfig(
                    prompt=PromptConfig(input_tokens=InputTokensConfig(mean=128))
                ),
            )

    def test_rejects_explicit_batch_size(self):
        with pytest.raises(ValidationError, match="--batch-size-text"):
            self._make_image_retrieval(
                input_config=InputConfig(prompt=PromptConfig(batch_size=4)),
            )

    def test_rejects_sequence_distribution(self):
        with pytest.raises(ValidationError, match="--sequence-distribution"):
            self._make_image_retrieval(
                input_config=InputConfig(
                    prompt=PromptConfig(sequence_distribution="128,64:50;256,128:50")
                ),
            )

    def test_rejects_prefix_prompt_options(self):
        with pytest.raises(ValidationError, match="Prefix prompt options"):
            self._make_image_retrieval(
                input_config=InputConfig(
                    prompt=PromptConfig(
                        prefix_prompt=PrefixPromptConfig(pool_size=5, length=100)
                    )
                ),
            )

    def test_allows_tokenizing_endpoint_with_text_options(self):
        config = make_config(
            endpoint=make_endpoint(endpoint_type=EndpointType.CHAT),
            input_config=InputConfig(
                prompt=PromptConfig(input_tokens=InputTokensConfig(mean=128))
            ),
        )
        assert config.input.prompt.input_tokens.mean == 128


class TestNonTokenEndpointTokenizerValidation:
    """Tests for reject_tokenizer_for_non_token_endpoints validator."""

    def test_rejects_explicit_tokenizer_name(self):
        with pytest.raises(ValidationError, match="Tokenizer options cannot be used"):
            make_config(
                endpoint=make_endpoint(endpoint_type=EndpointType.IMAGE_RETRIEVAL),
                tokenizer=TokenizerConfig(name="some-tokenizer"),
            )

    def test_rejects_explicit_tokenizer_revision(self):
        with pytest.raises(ValidationError, match="Tokenizer options cannot be used"):
            make_config(
                endpoint=make_endpoint(endpoint_type=EndpointType.IMAGE_RETRIEVAL),
                tokenizer=TokenizerConfig(revision="v2.0"),
            )

    def test_rejects_explicit_trust_remote_code(self):
        with pytest.raises(ValidationError, match="Tokenizer options cannot be used"):
            make_config(
                endpoint=make_endpoint(endpoint_type=EndpointType.IMAGE_RETRIEVAL),
                tokenizer=TokenizerConfig(trust_remote_code=True),
            )

    def test_allows_default_tokenizer_on_non_token_endpoint(self):
        config = make_config(
            endpoint=make_endpoint(endpoint_type=EndpointType.IMAGE_RETRIEVAL),
        )
        assert config.tokenizer.name is None

    def test_allows_tokenizer_on_tokenizing_endpoint(self):
        config = make_config(
            endpoint=make_endpoint(endpoint_type=EndpointType.CHAT),
            tokenizer=TokenizerConfig(name="some-tokenizer"),
        )
        assert config.tokenizer.name == "some-tokenizer"


class TestSweepIncompatibilitiesValidation:
    """Tests for parameter sweep incompatibility validation."""

    @patch("pathlib.Path.is_file", return_value=True)
    def test_sweep_with_fixed_schedule_raises_error(self, mock_is_file):
        """Test that parameter sweep with fixed_schedule raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            make_config(
                input_config=InputConfig(
                    fixed_schedule=True,
                    file="/tmp/trace.jsonl",
                ),
                loadgen=LoadGeneratorConfig(
                    concurrency=[10, 20, 30],
                    request_count=100,
                ),
            )

        error_msg = str(exc_info.value)
        assert "Parameter sweeps" in error_msg
        assert "cannot be used with --fixed-schedule mode" in error_msg

    @patch("pathlib.Path.is_file", return_value=True)
    def test_sweep_with_single_concurrency_and_fixed_schedule_succeeds(
        self, mock_is_file
    ):
        """Test that single concurrency with fixed_schedule succeeds."""
        config = make_config(
            input_config=InputConfig(
                fixed_schedule=True,
                file="/tmp/trace.jsonl",
            ),
            loadgen=LoadGeneratorConfig(
                concurrency=10,
                request_count=100,
            ),
        )
        assert config.loadgen.concurrency == 10
        assert config.input.fixed_schedule is True

    def test_sweep_without_fixed_schedule_succeeds(self):
        """Test that parameter sweep without fixed_schedule succeeds."""
        config = make_config(
            input_config=InputConfig(
                fixed_schedule=False,
            ),
            loadgen=LoadGeneratorConfig(
                concurrency=[10, 20, 30],
                request_count=100,
            ),
        )
        assert config.loadgen.concurrency == [10, 20, 30]
        assert config.input.fixed_schedule is False

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"timestamp": 1000, "text_input": "test", "output_length": 10}\n',
    )
    @patch("pathlib.Path.is_file", return_value=True)
    def test_sweep_with_mooncake_trace_with_timestamps_raises_error(
        self, mock_is_file, mock_file
    ):
        """Test that parameter sweep with mooncake_trace (with timestamps) raises ValueError."""
        from aiperf.plugin.enums import CustomDatasetType

        with pytest.raises(ValidationError) as exc_info:
            make_config(
                input_config=InputConfig(
                    custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
                    file="/tmp/trace.jsonl",
                ),
                loadgen=LoadGeneratorConfig(
                    concurrency=[10, 20, 30],
                    request_count=100,
                ),
            )

        error_msg = str(exc_info.value)
        assert "Parameter sweeps" in error_msg
        assert "mooncake_trace datasets" in error_msg
        assert "timestamps" in error_msg

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"text_input": "test", "output_length": 10}\n',
    )
    @patch("pathlib.Path.is_file", return_value=True)
    def test_sweep_with_mooncake_trace_without_timestamps_succeeds(
        self, mock_is_file, mock_file
    ):
        """Test that parameter sweep with mooncake_trace (without timestamps) succeeds."""
        from aiperf.plugin.enums import CustomDatasetType

        config = make_config(
            input_config=InputConfig(
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
                file="/tmp/trace.jsonl",
            ),
            loadgen=LoadGeneratorConfig(
                concurrency=[10, 20, 30],
                request_count=100,
            ),
        )
        assert config.loadgen.concurrency == [10, 20, 30]
        assert config.input.custom_dataset_type == CustomDatasetType.MOONCAKE_TRACE

    def test_sweep_with_multi_run_succeeds(self):
        """Test that parameter sweep with multi-run configuration succeeds."""
        config = make_config(
            loadgen=LoadGeneratorConfig(
                concurrency=[10, 20, 30],
                num_profile_runs=5,
                parameter_sweep_mode="repeated",
                request_count=100,
            ),
        )
        assert config.loadgen.concurrency == [10, 20, 30]
        assert config.loadgen.num_profile_runs == 5
        assert config.loadgen.parameter_sweep_mode == "repeated"

    def test_sweep_with_all_sweep_params_succeeds(self):
        """Test that parameter sweep with all sweep parameters succeeds."""
        config = make_config(
            loadgen=LoadGeneratorConfig(
                concurrency=[10, 20, 30],
                parameter_sweep_mode="independent",
                parameter_sweep_cooldown_seconds=5.0,
                parameter_sweep_same_seed=True,
                request_count=100,
            ),
        )
        assert config.loadgen.concurrency == [10, 20, 30]
        assert config.loadgen.parameter_sweep_mode == "independent"
        assert config.loadgen.parameter_sweep_cooldown_seconds == 5.0
        assert config.loadgen.parameter_sweep_same_seed is True

    def test_sweep_with_request_rate_succeeds(self):
        """Test that parameter sweep with request_rate succeeds."""
        config = make_config(
            loadgen=LoadGeneratorConfig(
                concurrency=[10, 20, 30],
                request_rate=100.0,
                request_count=100,
            ),
        )
        assert config.loadgen.concurrency == [10, 20, 30]
        assert config.loadgen.request_rate == 100.0

    def test_sweep_with_user_centric_rate_succeeds(self):
        """Test that parameter sweep with user_centric_rate succeeds."""
        config = make_config(
            input_config=make_multi_turn_input(num=50),
            loadgen=LoadGeneratorConfig(
                concurrency=[10, 20, 30],
                user_centric_rate=5.0,
                num_users=10,
            ),
        )
        assert config.loadgen.concurrency == [10, 20, 30]
        assert config.loadgen.user_centric_rate == 5.0
        assert config.loadgen.num_users == 10


class TestConcurrencyListParsing:
    """Tests for comma-separated concurrency list parsing (Task 2.1)."""

    def test_parse_concurrency_single_integer_string_returns_int(self):
        """Test parsing single integer string returns int (backward compatibility)."""
        config = LoadGeneratorConfig.model_validate({"concurrency": "10"})
        assert config.concurrency == 10
        assert isinstance(config.concurrency, int)

    def test_parse_concurrency_single_integer_returns_int(self):
        """Test single integer value remains as int (backward compatibility)."""
        config = LoadGeneratorConfig(concurrency=10)
        assert config.concurrency == 10
        assert isinstance(config.concurrency, int)

    def test_parse_concurrency_comma_separated_list_returns_list(self):
        """Test parsing comma-separated string returns list of integers."""
        config = LoadGeneratorConfig.model_validate({"concurrency": "10,20,30"})
        assert config.concurrency == [10, 20, 30]
        assert isinstance(config.concurrency, list)

    def test_parse_concurrency_with_spaces_returns_list(self):
        """Test parsing comma-separated string with spaces."""
        config = LoadGeneratorConfig.model_validate({"concurrency": "10, 20, 30, 40"})
        assert config.concurrency == [10, 20, 30, 40]

    def test_parse_concurrency_with_extra_spaces_returns_list(self):
        """Test parsing comma-separated string with extra whitespace."""
        config = LoadGeneratorConfig.model_validate({"concurrency": "10,  20,   30"})
        assert config.concurrency == [10, 20, 30]

    def test_parse_concurrency_list_of_integers_returns_list(self):
        """Test that list of integers passes through unchanged."""
        config = LoadGeneratorConfig(concurrency=[10, 20, 30])
        assert config.concurrency == [10, 20, 30]
        assert isinstance(config.concurrency, list)

    def test_parse_concurrency_none_value_returns_none(self):
        """Test that None value is preserved."""
        config = LoadGeneratorConfig(concurrency=None)
        assert config.concurrency is None

    def test_parse_concurrency_invalid_string_raises_error(self):
        """Test that invalid string value raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            LoadGeneratorConfig.model_validate({"concurrency": "abc"})
        assert "Invalid concurrency value" in str(exc_info.value)

    def test_parse_concurrency_list_with_non_integer_raises_error(self):
        """Test that comma-separated list with non-integer raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            LoadGeneratorConfig.model_validate({"concurrency": "10,abc,30"})
        assert "Invalid concurrency list" in str(exc_info.value)

    def test_parse_concurrency_duplicate_values_raises_error(self):
        """Test that duplicate values in list are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            LoadGeneratorConfig.model_validate({"concurrency": "10,20,10,30"})
        assert "Duplicate values" in str(exc_info.value)

    def test_parse_concurrency_empty_string_raises_error(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            LoadGeneratorConfig.model_validate({"concurrency": ""})
        assert "Invalid concurrency value" in str(exc_info.value)

    def test_parse_concurrency_trailing_comma_raises_error(self):
        """Test parsing single value with trailing comma."""
        # "10," splits into ["10", ""] - the empty string will cause an error
        with pytest.raises(ValidationError) as exc_info:
            LoadGeneratorConfig.model_validate({"concurrency": "10,"})
        assert "Invalid concurrency" in str(exc_info.value)

    def test_parse_concurrency_negative_single_value_raises_error(self):
        """Test that negative single value is rejected by validation."""
        with pytest.raises(ValidationError) as exc_info:
            LoadGeneratorConfig.model_validate({"concurrency": "-5"})
        assert "Invalid concurrency value" in str(
            exc_info.value
        ) or "must be >= 1" in str(exc_info.value)

    def test_parse_concurrency_zero_single_value_raises_error(self):
        """Test that zero single value is rejected by validation."""
        with pytest.raises(ValidationError) as exc_info:
            LoadGeneratorConfig.model_validate({"concurrency": "0"})
        assert "Invalid concurrency value" in str(
            exc_info.value
        ) or "must be >= 1" in str(exc_info.value)

    def test_validation_rejects_negative_in_list(self):
        """Test that negative value in list is rejected by validation."""
        with pytest.raises(ValidationError) as exc_info:
            LoadGeneratorConfig.model_validate({"concurrency": "10,-5,30"})
        assert "Invalid concurrency" in str(
            exc_info.value
        ) and "must be positive integers (>= 1)" in str(exc_info.value)

    def test_validation_rejects_zero_in_list(self):
        """Test that zero value in list is rejected by validation."""
        with pytest.raises(ValidationError) as exc_info:
            LoadGeneratorConfig.model_validate({"concurrency": "10,0,30"})
        assert "Invalid concurrency" in str(
            exc_info.value
        ) and "must be positive integers (>= 1)" in str(exc_info.value)

    def test_parse_concurrency_large_values_returns_list(self):
        """Test parsing large concurrency values."""
        config = LoadGeneratorConfig.model_validate({"concurrency": "100,500,1000"})
        assert config.concurrency == [100, 500, 1000]

    def test_parse_concurrency_many_values_returns_list(self):
        """Test parsing many concurrency values."""
        values = ",".join(str(i * 10) for i in range(1, 11))
        config = LoadGeneratorConfig.model_validate({"concurrency": values})
        assert config.concurrency == [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    def test_parse_concurrency_integration_with_loadgen_config(self):
        """Test that concurrency list parsing works in LoadGeneratorConfig."""
        config = LoadGeneratorConfig.model_validate(
            {"concurrency": "10,20,30", "request_count": 100}
        )
        assert config.concurrency == [10, 20, 30]
        assert config.request_count == 100
