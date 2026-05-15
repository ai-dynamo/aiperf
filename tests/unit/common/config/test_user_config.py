# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the v1 CLIConfig DTO.

The v1 CLIConfig is a CLI-only input DTO. Validators are forbidden on it
(see `aiperf.config.flags.cli_config` module docstring) - all the cross-field
domain rules that older revisions of this file asserted (concurrency vs
request_count, num_users / user_centric_rate / arrival_pattern coupling,
prefill_concurrency streaming requirement, GPU-telemetry pynvml-vs-DCGM
exclusivity, sweep-vs-fixed-schedule, rankings options gated on endpoint,
non-tokenizing endpoints rejecting tokenizer/prefix-prompt knobs, etc.)
moved to AIPerfConfig in v2. The relevant tests now live alongside the v2
config (``tests/unit/config/test_end_to_end_config_flow.py`` and friends).

What survives here:

* Round-trip serialization (model_dump_json/model_validate_json) with the
  flattened endpoint section.
* Default smoke tests that the nested DTOs are constructed.
"""

from unittest.mock import mock_open, patch

from aiperf.config.endpoint import EndpointDefaults
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.plugin.enums import (
    DatasetSamplingStrategy,
    EndpointType,
)


def make_endpoint_kwargs(
    endpoint_type: EndpointType = EndpointType.CHAT,
    streaming: bool = False,
    model_names: list[str] | None = None,
    **kwargs,
) -> dict:
    """Return endpoint-section kwargs to spread into a CLIConfig(...) call."""
    if "url" in kwargs:
        kwargs["urls"] = [kwargs.pop("url")]
    return {
        "model_names": model_names or ["test-model"],
        "type": endpoint_type,
        "custom_endpoint": kwargs.pop("custom_endpoint", "test"),
        "streaming": streaming,
        **kwargs,
    }


def make_config(
    endpoint_kwargs: dict | None = None,
    loadgen: CLIConfig | None = None,
    **kwargs,
) -> CLIConfig:
    config_kwargs = {**(endpoint_kwargs or make_endpoint_kwargs()), **kwargs}
    if loadgen is not None:
        config_kwargs.update(loadgen.model_dump(exclude_unset=True))
    return CLIConfig(**config_kwargs)


class TestCLIConfigSerialization:
    """Tests for CLIConfig serialization and deserialization."""

    def test_to_json_string(self):
        """Round-trip a populated CLIConfig through JSON."""
        config = CLIConfig(
            model_names=["model1", "model2"],
            endpoint_type=EndpointType.CHAT,
            custom_endpoint="custom_endpoint",
            streaming=True,
            urls=["http://custom-url"],
            api_key="test_api_key",
            timeout_seconds=10,
            random_seed=42,
            dataset_sampling_strategy=DatasetSamplingStrategy.SHUFFLE,
            extra_inputs=[("key1", "value1"), ("key2", "value2"), ("key3", "value3")],
            headers=[
                ("Authorization", "Bearer token"),
                ("Content-Type", "application/json"),
            ],
            conversation_num=10,
            conversation_turn_mean=10,
            conversation_turn_stddev=10,
            conversation_turn_delay_mean=10,
            conversation_turn_delay_stddev=10,
            artifact_directory="test_artifacts",
            tokenizer_name="test_tokenizer",
            tokenizer_revision="test_revision",
            **CLIConfig(concurrency=10, request_rate=10).model_dump(exclude_unset=True),
            cli_command="test_cli_command",
        )

        ctx = {"include_secrets": True}
        assert (
            CLIConfig.model_validate_json(
                config.model_dump_json(indent=4, exclude_unset=True, context=ctx)
            )
            == config
        )
        assert (
            CLIConfig.model_validate_json(
                config.model_dump_json(indent=4, exclude_defaults=True, context=ctx)
            )
            == config
        )

    def test_to_file(self):
        """Round-trip via a (mocked) file."""
        config = make_config(
            endpoint_kwargs=make_endpoint_kwargs(
                streaming=True, url="http://custom-url"
            ),
        )

        mocked_file = mock_open()
        with patch("pathlib.Path.open", mocked_file):
            mocked_file().write(config.model_dump_json(indent=4, exclude_defaults=True))

        with patch("pathlib.Path.open", mocked_file):
            mocked_file().read.return_value = config.model_dump_json(
                indent=4, exclude_defaults=True
            )
            loaded_config = CLIConfig.model_validate_json(mocked_file().read())

        assert config == loaded_config

    def test_exclude_unset_fields(self):
        """Selecting various exclude flags changes the JSON output."""
        config = make_config(
            endpoint_kwargs=make_endpoint_kwargs(
                streaming=True, url="http://custom-url"
            ),
        )
        assert config.model_dump_json(exclude_unset=True) != config.model_dump_json()
        assert config.model_dump_json(exclude_defaults=True) != config.model_dump_json()
        assert (
            config.model_dump_json(exclude_unset=True, exclude_defaults=True)
            != config.model_dump_json()
        )
        assert config.model_dump_json(exclude_none=True) != config.model_dump_json()


class TestCLIConfigDefaults:
    """Tests for nested-DTO defaulting on CLIConfig."""

    def test_defaults(self):
        config = make_config(
            endpoint_kwargs=make_endpoint_kwargs(model_names=["model1", "model2"])
        )

        assert config.model_names == ["model1", "model2"]
        assert config.streaming == EndpointDefaults.STREAMING
        assert config.url == EndpointDefaults.URL
        # Modality fields are flat top-level on CLIConfig post-Task-13.
        assert config.audio_batch_size == 1
        # Output fields hoisted to top-level CLIConfig (no nested section).
        # Tokenizer fields hoisted to top-level CLIConfig (no nested section).
        assert config.tokenizer_name is None
        assert config.tokenizer_revision == "main"
        assert config.trust_remote_code is False

    def test_custom_values(self):
        config = make_config(
            endpoint_kwargs=make_endpoint_kwargs(
                model_names=["model1", "model2"],
                streaming=True,
                url="http://custom-url",
            ),
            loadgen=CLIConfig(concurrency=4),
        )

        assert config.model_names == ["model1", "model2"]
        assert config.streaming is True
        assert config.url == "http://custom-url"
        assert config.concurrency == 4
