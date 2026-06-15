# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for record payload-bytes retention auto-detection.

These build REAL ``UserConfig`` / ``ModelEndpointInfo`` objects (not mocks) so
the attribute paths the predicate reads are validated against the actual config
schema -- a MagicMock would auto-create whatever path we ask for and hide drift.
"""

import pytest
from pytest import param

from aiperf.common.config import UserConfig
from aiperf.common.environment import Environment
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.plugin import plugins
from aiperf.records.payload_retention import (
    record_payload_bytes_required,
    resolve_disable_tokenization,
    resolve_strip_record_payload_bytes,
)


def _make_user_config(
    *,
    use_server_token_count: bool = False,
    export_level: str = "records",
    image: bool = False,
    audio: bool = False,
    video: bool = False,
    endpoint_type: str = "chat",
) -> UserConfig:
    """Build a real UserConfig with the signals the predicate reads."""
    input_cfg: dict = {}
    if image:
        input_cfg["image"] = {"width": {"mean": 64}, "height": {"mean": 64}}
    if audio:
        input_cfg["audio"] = {"length": {"mean": 1.0}}
    if video:
        input_cfg["video"] = {"width": 64, "height": 64}
    return UserConfig(
        endpoint={
            "model_names": ["test-model"],
            "type": endpoint_type,
            "use_server_token_count": use_server_token_count,
        },
        output={"export_level": export_level},
        input=input_cfg,
    )


def _model_endpoint(user_config: UserConfig) -> ModelEndpointInfo:
    return ModelEndpointInfo.from_user_config(user_config)


class TestResolveDisableTokenization:
    """resolve_disable_tokenization mirrors the parser's derivation against
    REAL endpoint plugin metadata."""

    def test_chat_with_client_tokenization_is_enabled(self):
        uc = _make_user_config(use_server_token_count=False)
        meta = plugins.get_endpoint_metadata("chat")
        # chat both produces and tokenizes -> client-side tokenization runs.
        assert resolve_disable_tokenization(uc, meta) is False

    def test_server_token_count_disables_tokenization(self):
        uc = _make_user_config(use_server_token_count=True)
        meta = plugins.get_endpoint_metadata("chat")
        assert resolve_disable_tokenization(uc, meta) is True


class TestRecordPayloadBytesRequired:
    """The predicate is True iff some downstream consumer reads payload_bytes."""

    def test_text_only_server_tokens_no_export_is_not_required(self):
        """The canonical strippable run: server token counts, text-only,
        records (non-raw) export -> nothing reads payload_bytes."""
        uc = _make_user_config(use_server_token_count=True)
        assert record_payload_bytes_required(uc, _model_endpoint(uc)) is False

    def test_client_side_tokenization_requires_payload(self):
        uc = _make_user_config(use_server_token_count=False)
        assert record_payload_bytes_required(uc, _model_endpoint(uc)) is True

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
        uc = _make_user_config(use_server_token_count=True, **media_kwargs)
        assert record_payload_bytes_required(uc, _model_endpoint(uc)) is True

    def test_raw_export_requires_payload(self):
        uc = _make_user_config(use_server_token_count=True, export_level="raw")
        assert record_payload_bytes_required(uc, _model_endpoint(uc)) is True


class TestResolveStripRecordPayloadBytes:
    """Tri-state resolution: None auto-detects, True/False override."""

    def test_none_auto_strips_when_not_required(self, monkeypatch):
        monkeypatch.setattr(Environment.RECORD, "STRIP_PAYLOAD_BYTES", None)
        uc = _make_user_config(use_server_token_count=True)  # predicate False
        assert resolve_strip_record_payload_bytes(uc, _model_endpoint(uc)) is True

    def test_none_keeps_when_required(self, monkeypatch):
        monkeypatch.setattr(Environment.RECORD, "STRIP_PAYLOAD_BYTES", None)
        uc = _make_user_config(use_server_token_count=False)  # predicate True
        assert resolve_strip_record_payload_bytes(uc, _model_endpoint(uc)) is False

    def test_explicit_true_forces_strip_even_when_required(self, monkeypatch):
        monkeypatch.setattr(Environment.RECORD, "STRIP_PAYLOAD_BYTES", True)
        uc = _make_user_config(use_server_token_count=False)  # predicate True
        assert resolve_strip_record_payload_bytes(uc, _model_endpoint(uc)) is True

    def test_explicit_false_forces_keep_even_when_not_required(self, monkeypatch):
        monkeypatch.setattr(Environment.RECORD, "STRIP_PAYLOAD_BYTES", False)
        uc = _make_user_config(use_server_token_count=True)  # predicate False
        assert resolve_strip_record_payload_bytes(uc, _model_endpoint(uc)) is False


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

        uc = _make_user_config(use_server_token_count=True)
        me = _model_endpoint(uc)
        assert record_payload_bytes_required(uc, me) is False

        # Consumer 1: client-side input tokenization (parser delegates to the
        # same resolve_disable_tokenization function).
        meta = plugins.get_endpoint_metadata(me.endpoint.type)
        assert resolve_disable_tokenization(uc, meta) is True
        # Consumer 2: media counting from request bodies.
        assert _run_has_synthetic_media(uc) is False
        # Consumer 3: raw payload export.
        assert uc.output.export_level != ExportLevel.RAW
