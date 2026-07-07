# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for record client-side tokenization derivation.

These build REAL ``BenchmarkConfig`` / ``ModelEndpointInfo`` objects (not mocks)
so the attribute paths the predicate reads are validated against the actual
config schema -- a MagicMock would auto-create whatever path we ask for and hide
drift. The v2 payload-retention predicates take a ``BenchmarkConfig`` (not the
v1 ``UserConfig``); a real ``BenchmarkRun`` is built via the resolver so its
``cfg`` carries the resolved endpoint/output/synthetic-media shapes.
"""

from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.plugin import plugins
from aiperf.records.payload_retention import resolve_disable_tokenization
from tests.unit.conftest import make_run_from_cli


def _make_run(
    *,
    use_server_token_count: bool = False,
    export_level: str = "records",
    image: bool = False,
    audio: bool = False,
    video: bool = False,
    endpoint_type: str = "chat",
):
    """Build a real v2 BenchmarkRun with the signals the predicate reads."""
    overrides: dict = {
        "model_names": ["test-model"],
        "endpoint_type": endpoint_type,
        "use_server_token_count": use_server_token_count,
        "export_level": export_level,
    }
    if image:
        overrides["image_width_mean"] = 64
        overrides["image_height_mean"] = 64
    if audio:
        overrides["audio_length_mean"] = 1.0
    if video:
        overrides["video_width"] = 64
        overrides["video_height"] = 64
    return make_run_from_cli(CLIConfig(**overrides))


def _model_endpoint(run) -> ModelEndpointInfo:
    return ModelEndpointInfo.from_run(run)


class TestResolveDisableTokenization:
    """resolve_disable_tokenization mirrors the parser's derivation against
    REAL endpoint plugin metadata."""

    def test_chat_with_client_tokenization_is_enabled(self):
        uc = _make_run(use_server_token_count=False)
        meta = plugins.get_endpoint_metadata("chat")
        # chat both produces and tokenizes -> client-side tokenization runs.
        assert resolve_disable_tokenization(uc.cfg, meta) is False

    def test_server_token_count_disables_tokenization(self):
        uc = _make_run(use_server_token_count=True)
        meta = plugins.get_endpoint_metadata("chat")
        assert resolve_disable_tokenization(uc.cfg, meta) is True
