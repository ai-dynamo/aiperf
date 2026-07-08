# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for record payload-bytes retention auto-detection.

These build REAL ``BenchmarkConfig`` / ``ModelEndpointInfo`` objects (not mocks)
so the attribute paths the predicate reads are validated against the actual
config schema -- a MagicMock would auto-create whatever path we ask for and hide
drift. The v2 payload-retention predicates take a ``BenchmarkConfig`` (not the
v1 ``UserConfig``); a real ``BenchmarkRun`` is built via the resolver so its
``cfg`` carries the resolved endpoint/output/synthetic-media shapes.
"""

import pytest
from pytest import param

from aiperf.common.environment import Environment
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.plugin import plugins
from aiperf.records.payload_retention import (
    record_payload_bytes_required,
    resolve_disable_tokenization,
    resolve_strip_record_payload_bytes,
)
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


class TestRecordPayloadBytesRequired:
    """The predicate is True iff some downstream consumer reads payload_bytes."""

    def test_text_only_server_tokens_no_export_is_not_required(self):
        """The canonical strippable run: server token counts, text-only,
        records (non-raw) export -> nothing reads payload_bytes."""
        uc = _make_run(use_server_token_count=True)
        assert record_payload_bytes_required(uc.cfg, _model_endpoint(uc)) is False

    def test_client_side_tokenization_requires_payload(self):
        uc = _make_run(use_server_token_count=False)
        assert record_payload_bytes_required(uc.cfg, _model_endpoint(uc)) is True

    @pytest.mark.parametrize(
        "media_kwargs",
        [
            param({"image": True}, id="image"),
            param({"audio": True}, id="audio"),
            param({"video": True}, id="video"),
        ],
    )
    def test_synthetic_media_requires_payload(self, media_kwargs):
        """Media counts derive from the request body, so configured synthetic
        media keeps payload_bytes even under server token counts."""
        uc = _make_run(use_server_token_count=True, **media_kwargs)
        assert record_payload_bytes_required(uc.cfg, _model_endpoint(uc)) is True

    def test_raw_export_requires_payload(self):
        uc = _make_run(use_server_token_count=True, export_level="raw")
        assert record_payload_bytes_required(uc.cfg, _model_endpoint(uc)) is True


class TestResolveStripRecordPayloadBytes:
    """Tri-state resolution: None auto-detects, True/False override."""

    def test_none_auto_strips_when_not_required(self, monkeypatch):
        monkeypatch.setattr(Environment.RECORD, "STRIP_PAYLOAD_BYTES", None)
        uc = _make_run(use_server_token_count=True)  # predicate False
        assert resolve_strip_record_payload_bytes(uc.cfg, _model_endpoint(uc)) is True

    def test_none_keeps_when_required(self, monkeypatch):
        monkeypatch.setattr(Environment.RECORD, "STRIP_PAYLOAD_BYTES", None)
        uc = _make_run(use_server_token_count=False)  # predicate True
        assert resolve_strip_record_payload_bytes(uc.cfg, _model_endpoint(uc)) is False

    def test_explicit_true_forces_strip_even_when_required(self, monkeypatch):
        monkeypatch.setattr(Environment.RECORD, "STRIP_PAYLOAD_BYTES", True)
        uc = _make_run(use_server_token_count=False)  # predicate True
        assert resolve_strip_record_payload_bytes(uc.cfg, _model_endpoint(uc)) is True

    def test_explicit_false_forces_keep_even_when_not_required(self, monkeypatch):
        monkeypatch.setattr(Environment.RECORD, "STRIP_PAYLOAD_BYTES", False)
        uc = _make_run(use_server_token_count=True)  # predicate False
        assert resolve_strip_record_payload_bytes(uc.cfg, _model_endpoint(uc)) is False


class TestAutoStripConsumerGuard:
    """Anti-drift guard: when auto-detection chooses to strip, every known
    payload_bytes consumer must be provably inert for that run. If a future
    consumer starts reading payload_bytes, add its gate to
    record_payload_bytes_required (and assert it here), or auto-strip will
    silently feed it None.
    """

    def test_strippable_run_has_all_consumers_inert(self):
        from aiperf.common.enums import ExportLevel
        from aiperf.records.payload_retention import _run_has_synthetic_media

        uc = _make_run(use_server_token_count=True)
        me = _model_endpoint(uc)
        assert record_payload_bytes_required(uc.cfg, me) is False

        # Consumer 1: client-side input tokenization (parser delegates to the
        # same resolve_disable_tokenization function).
        meta = plugins.get_endpoint_metadata(me.endpoint.type)
        assert resolve_disable_tokenization(uc.cfg, meta) is True
        # Consumer 2: media counting from request bodies.
        assert _run_has_synthetic_media(uc.cfg) is False
        # Consumer 3: raw payload export.
        assert uc.cfg.artifacts.export_level != ExportLevel.RAW
