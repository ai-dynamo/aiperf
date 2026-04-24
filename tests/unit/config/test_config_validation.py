# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for cross-section validators on AIPerfConfig."""

from __future__ import annotations

import warnings

import pytest
from pydantic import ValidationError

from aiperf.config.config import AIPerfConfig
from aiperf.config.endpoint import EndpointConfig


def _base_config(**overrides) -> dict:
    """Minimal valid AIPerfConfig dict with overrides."""
    base = {
        "models": ["test-model"],
        "endpoint": {
            "urls": ["http://localhost:8000/v1/chat/completions"],
        },
        "datasets": {
            "main": {
                "type": "synthetic",
                "entries": 100,
                "prompts": {"isl": 128, "osl": 64},
            },
        },
        "phases": {
            "profiling": {
                "type": "concurrency",
                "concurrency": 8,
                "requests": 100,
            },
        },
    }
    base.update(overrides)
    return base


# =============================================================================
# GAP #2: prefill_concurrency requires streaming (ForEach)
# =============================================================================


class TestPrefillConcurrencyRequiresStreaming:
    """ForEach wired constraint: prefill_concurrency requires endpoint.streaming."""

    def test_prefill_with_streaming_passes(self):
        config = AIPerfConfig(
            **_base_config(
                endpoint={
                    "urls": ["http://localhost:8000/v1/chat/completions"],
                    "streaming": True,
                },
                phases={
                    "profiling": {
                        "type": "concurrency",
                        "concurrency": 8,
                        "prefill_concurrency": 4,
                        "requests": 100,
                    },
                },
            )
        )
        assert config.phases["profiling"].prefill_concurrency == 4
        assert config.endpoint.streaming is True

    def test_prefill_without_streaming_fails(self):
        with pytest.raises(
            ValidationError, match="prefill_concurrency requires endpoint.streaming"
        ):
            AIPerfConfig(
                **_base_config(
                    endpoint={
                        "urls": ["http://localhost:8000/v1/chat/completions"],
                        "streaming": False,
                    },
                    phases={
                        "profiling": {
                            "type": "concurrency",
                            "concurrency": 8,
                            "prefill_concurrency": 4,
                            "requests": 100,
                        },
                    },
                )
            )

    def test_no_prefill_without_streaming_passes(self):
        config = AIPerfConfig(
            **_base_config(
                endpoint={
                    "urls": ["http://localhost:8000/v1/chat/completions"],
                    "streaming": False,
                },
            )
        )
        assert config.endpoint.streaming is False

    def test_multiple_phases_one_fails(self):
        with pytest.raises(
            ValidationError, match="prefill_concurrency requires endpoint.streaming"
        ):
            AIPerfConfig(
                **_base_config(
                    endpoint={
                        "urls": ["http://localhost:8000/v1/chat/completions"],
                        "streaming": False,
                    },
                    phases={
                        "warmup": {
                            "type": "concurrency",
                            "concurrency": 4,
                            "requests": 50,
                            "exclude_from_results": True,
                        },
                        "profiling": {
                            "type": "concurrency",
                            "concurrency": 8,
                            "prefill_concurrency": 4,
                            "requests": 100,
                        },
                    },
                )
            )

    def test_multiple_phases_all_pass(self):
        config = AIPerfConfig(
            **_base_config(
                endpoint={
                    "urls": ["http://localhost:8000/v1/chat/completions"],
                    "streaming": True,
                },
                phases={
                    "warmup": {
                        "type": "concurrency",
                        "concurrency": 4,
                        "prefill_concurrency": 2,
                        "requests": 50,
                        "exclude_from_results": True,
                    },
                    "profiling": {
                        "type": "concurrency",
                        "concurrency": 8,
                        "prefill_concurrency": 4,
                        "requests": 100,
                    },
                },
            )
        )
        assert len(config.phases) == 2


# =============================================================================
# GAP #9: api_host requires api_port (F() wired constraint)
# =============================================================================


class TestApiHostRequiresApiPort:
    """F() wired constraint: runtime.api_host requires runtime.api_port."""

    def test_both_set_passes(self):
        config = AIPerfConfig(
            **_base_config(
                runtime={
                    "api_host": "0.0.0.0",
                    "api_port": 9090,
                },
            )
        )
        assert config.runtime.api_host == "0.0.0.0"
        assert config.runtime.api_port == 9090

    def test_host_without_port_fails(self):
        with pytest.raises(ValidationError, match="api_port"):
            AIPerfConfig(
                **_base_config(
                    runtime={
                        "api_host": "0.0.0.0",
                    },
                )
            )

    def test_port_without_host_passes(self):
        config = AIPerfConfig(
            **_base_config(
                runtime={
                    "api_port": 9090,
                },
            )
        )
        assert config.runtime.api_port == 9090
        assert config.runtime.api_host is None

    def test_neither_set_passes(self):
        config = AIPerfConfig(**_base_config())
        assert config.runtime.api_host is None
        assert config.runtime.api_port is None


# =============================================================================
# GAP #6: fixed_schedule requires sequential sampling
# =============================================================================


class TestFixedScheduleSampling:
    """fixed_schedule phases require sequential sampling on their dataset."""

    def test_fixed_schedule_with_sequential_passes(self):
        config = AIPerfConfig(
            **_base_config(
                datasets={
                    "trace": {
                        "type": "file",
                        "path": "/tmp/trace.jsonl",
                        "format": "mooncake_trace",
                        "sampling": "sequential",
                    },
                },
                phases={
                    "replay": {
                        "type": "fixed_schedule",
                        "dataset": "trace",
                    },
                },
            )
        )
        assert config.phases["replay"].type == "fixed_schedule"

    def test_fixed_schedule_with_random_fails(self):
        with pytest.raises(
            ValidationError, match="fixed_schedule.*sequential.*sampling"
        ):
            AIPerfConfig(
                **_base_config(
                    datasets={
                        "trace": {
                            "type": "file",
                            "path": "/tmp/trace.jsonl",
                            "format": "mooncake_trace",
                            "sampling": "random",
                        },
                    },
                    phases={
                        "replay": {
                            "type": "fixed_schedule",
                            "dataset": "trace",
                        },
                    },
                )
            )

    def test_fixed_schedule_with_synthetic_dataset_passes(self):
        """Synthetic datasets don't have format, so no validation needed."""
        config = AIPerfConfig(
            **_base_config(
                phases={
                    "replay": {
                        "type": "fixed_schedule",
                        "dataset": "main",
                    },
                },
            )
        )
        assert config.phases["replay"].type == "fixed_schedule"


# =============================================================================
# GAP #7: user_centric requires multi_turn
# =============================================================================


class TestUserCentricRequiresMultiTurn:
    """user_centric phases require multi_turn dataset format."""

    def test_user_centric_with_multi_turn_passes(self):
        config = AIPerfConfig(
            **_base_config(
                datasets={
                    "conversations": {
                        "type": "file",
                        "path": "/tmp/conversations.jsonl",
                        "format": "multi_turn",
                    },
                },
                phases={
                    "profiling": {
                        "type": "user_centric",
                        "dataset": "conversations",
                        "rate": 10.0,
                        "users": 5,
                        "requests": 100,
                    },
                },
            )
        )
        assert config.phases["profiling"].type == "user_centric"

    def test_user_centric_with_single_turn_fails(self):
        with pytest.raises(ValidationError, match="user_centric.*multi_turn.*format"):
            AIPerfConfig(
                **_base_config(
                    datasets={
                        "data": {
                            "type": "file",
                            "path": "/tmp/data.jsonl",
                            "format": "single_turn",
                        },
                    },
                    phases={
                        "profiling": {
                            "type": "user_centric",
                            "dataset": "data",
                            "rate": 10.0,
                            "users": 5,
                            "requests": 100,
                        },
                    },
                )
            )

    def test_user_centric_with_synthetic_passes(self):
        """Synthetic datasets aren't FileDataset, so no format check."""
        config = AIPerfConfig(
            **_base_config(
                phases={
                    "profiling": {
                        "type": "user_centric",
                        "dataset": "main",
                        "rate": 10.0,
                        "users": 5,
                        "requests": 100,
                    },
                },
            )
        )
        assert config.phases["profiling"].type == "user_centric"


# =============================================================================
# Streaming normalization for unsupported endpoint types
# =============================================================================


class TestStreamingNormalization:
    """EndpointConfig silently disables streaming for types that don't support it."""

    def test_streaming_true_with_chat_endpoint_preserved(self):
        endpoint = EndpointConfig(
            urls=["http://localhost:8000/v1/chat/completions"],
            streaming=True,
        )
        assert endpoint.streaming is True

    def test_streaming_false_with_chat_endpoint_preserved(self):
        endpoint = EndpointConfig(
            urls=["http://localhost:8000/v1/chat/completions"],
            streaming=False,
        )
        assert endpoint.streaming is False

    def test_streaming_true_with_embeddings_disabled_with_warning(self):
        from aiperf.config.endpoint import EndpointConfig

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            endpoint = EndpointConfig(
                urls=["http://localhost:8000/v1/embeddings"],
                type="embeddings",
                streaming=True,
            )

        assert endpoint.streaming is False
        assert any("streaming" in str(w.message).lower() for w in caught)

    def test_streaming_false_with_embeddings_no_warning(self):
        from aiperf.config.endpoint import EndpointConfig

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            endpoint = EndpointConfig(
                urls=["http://localhost:8000/v1/embeddings"],
                type="embeddings",
                streaming=False,
            )

        assert endpoint.streaming is False
        assert not any("streaming" in str(w.message).lower() for w in caught)

    def test_streaming_normalization_via_aiperf_config(self):
        """streaming=True on an embeddings endpoint is normalized to False in AIPerfConfig."""

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            config = AIPerfConfig(
                **_base_config(
                    endpoint={
                        "urls": ["http://localhost:8000/v1/embeddings"],
                        "type": "embeddings",
                        "streaming": True,
                    }
                )
            )
        assert config.endpoint.streaming is False
