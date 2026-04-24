# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from aiperf.config import AIPerfConfig, BenchmarkRun


@pytest.fixture(autouse=True)
def mock_image_loading():
    """Mock image loading for all composer tests to avoid filesystem dependencies."""
    with (
        patch("aiperf.dataset.generator.image.glob.glob") as mock_glob,
        patch("aiperf.dataset.generator.image.Image.open") as mock_open,
    ):
        # Return a fake image path
        mock_glob.return_value = ["/fake/path/test_image.jpg"]

        # Create a mock image with copy() method
        mock_image = Mock(spec=Image.Image)
        mock_image.copy.return_value = mock_image

        # Support context manager protocol
        mock_open.return_value.__enter__ = Mock(return_value=mock_image)
        mock_open.return_value.__exit__ = Mock(return_value=None)

        yield


@pytest.fixture
def mock_tokenizer(mock_tokenizer_cls):
    """Mock tokenizer class."""
    return mock_tokenizer_cls.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    )


# ============================================================================
# Base config helpers
# ============================================================================

_BASE = dict(
    models=["test-model"],
    endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
    phases={"default": {"type": "concurrency", "requests": 10, "concurrency": 1}},
)


def _make_run(config: AIPerfConfig) -> BenchmarkRun:
    """Wrap an AIPerfConfig in a BenchmarkRun for testing."""
    return BenchmarkRun(
        benchmark_id="test",
        cfg=config,
        artifact_dir=Path("/tmp/test"),
    )


def _config(**dataset_overrides) -> AIPerfConfig:
    """Build an AIPerfConfig with a single synthetic dataset, merging overrides."""
    dataset = {"type": "synthetic", "entries": 100, "prompts": {"isl": 128, "osl": 64}}
    dataset.update(dataset_overrides)
    return AIPerfConfig(**_BASE, datasets={"default": dataset})


# ============================================================================
# Synthetic Composer Fixtures
# ============================================================================


@pytest.fixture
def synthetic_config() -> BenchmarkRun:
    """Basic synthetic configuration for testing."""
    return _make_run(
        _config(
            entries=5,
            prompts={"isl": {"mean": 10, "stddev": 2}, "osl": 64},
        )
    )


@pytest.fixture
def image_config() -> BenchmarkRun:
    """Synthetic configuration with image generation enabled."""
    return _make_run(
        _config(
            entries=3,
            prompts={"isl": {"mean": 10, "stddev": 2}, "osl": 64},
            images={
                "batch_size": 1,
                "width": {"mean": 10},
                "height": {"mean": 10},
            },
        )
    )


@pytest.fixture
def audio_config() -> BenchmarkRun:
    """Synthetic configuration with audio generation enabled."""
    return _make_run(
        _config(
            entries=3,
            prompts={"isl": {"mean": 10, "stddev": 2}, "osl": 64},
            audio={"batch_size": 1, "length": {"mean": 2}},
        )
    )


@pytest.fixture
def prefix_prompt_config() -> BenchmarkRun:
    """Synthetic configuration with prefix prompts enabled."""
    return _make_run(
        _config(
            entries=5,
            prompts={"isl": {"mean": 10, "stddev": 2}, "osl": 64},
            prefix_prompts={"pool_size": 3, "length": 20},
        )
    )


@pytest.fixture
def multimodal_config() -> BenchmarkRun:
    """Synthetic configuration with multimodal data generation enabled."""
    return _make_run(
        _config(
            entries=2,
            prompts={
                "isl": {"mean": 10, "stddev": 2},
                "osl": 64,
                "batch_size": 2,
            },
            prefix_prompts={"pool_size": 2, "length": 15},
            images={
                "batch_size": 2,
                "width": {"mean": 10},
                "height": {"mean": 10},
            },
            audio={"batch_size": 2, "length": {"mean": 2}},
        )
    )


@pytest.fixture
def multiturn_config() -> BenchmarkRun:
    """Synthetic configuration with multiturn settings."""
    return _make_run(
        _config(
            entries=4,
            prompts={"isl": {"mean": 10, "stddev": 2}, "osl": 64},
            turns={"mean": 2, "stddev": 0},
            turn_delay={"mean": 1500, "stddev": 0},
        )
    )


# ============================================================================
# Custom Composer Fixtures
# ============================================================================


@pytest.fixture
def custom_config() -> BenchmarkRun:
    """Basic custom configuration for testing."""
    return _make_run(
        AIPerfConfig(
            **_BASE,
            datasets={
                "default": {
                    "type": "file",
                    "path": "test_data.jsonl",
                    "format": "single_turn",
                }
            },
        )
    )


@pytest.fixture
def trace_config() -> BenchmarkRun:
    """Configuration for TRACE dataset type."""
    return _make_run(
        AIPerfConfig(
            **_BASE,
            datasets={
                "default": {
                    "type": "file",
                    "path": "trace_data.jsonl",
                    "format": "mooncake_trace",
                }
            },
        )
    )
