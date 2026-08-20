# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record processors must load the tokenizer the WGM already downloaded.

The WorkerGroupManager prefetches tokenizer bundles once per pod and announces
them with GroupTokenizerReady. That producer survived the port; the consumer
did not, so every record processor fell back to fetching from the hub itself --
N duplicate downloads per pod, and a hard failure on an air-gapped or
gated-repo cluster where the WGM's copy was the only one obtainable.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.pod_lifecycle_structs import GroupTokenizerReady


@pytest.fixture
def rp():
    from aiperf.records.record_processor_service import RecordProcessor

    svc = RecordProcessor.__new__(RecordProcessor)
    svc.service_id = "record_processor_0"
    svc._tokenizer_bundles = {}
    svc._tokenizer_ready = __import__("asyncio").Event()
    svc.info = MagicMock()
    svc.error = MagicMock()
    svc.warning = MagicMock()
    svc.debug = MagicMock()
    svc.pod_lifecycle_dealer_client = AsyncMock()
    svc.inference_result_parser = MagicMock(_tokenizer_bundles={})
    return svc


class TestTokenizerReady:
    @pytest.mark.asyncio
    async def test_bundles_are_recorded_and_waiters_released(self, rp):
        await rp._on_tokenizer_ready(
            GroupTokenizerReady(
                service_id="wgm-0",
                success=True,
                bundles={"meta-llama/Llama-3-8B": "/pod/tok/llama"},
            )
        )
        assert rp._tokenizer_bundles == {"meta-llama/Llama-3-8B": "/pod/tok/llama"}
        assert rp._tokenizer_ready.is_set()

    @pytest.mark.asyncio
    async def test_further_announcements_merge(self, rp):
        await rp._on_tokenizer_ready(
            GroupTokenizerReady(service_id="wgm-0", success=True, bundles={"a": "/a"})
        )
        await rp._on_tokenizer_ready(
            GroupTokenizerReady(service_id="wgm-0", success=True, bundles={"b": "/b"})
        )
        assert rp._tokenizer_bundles == {"a": "/a", "b": "/b"}

    @pytest.mark.asyncio
    async def test_download_failure_hard_exits(self, rp, monkeypatch):
        """A pod whose tokenizer never arrives can never process a record."""
        exits: list[int] = []
        monkeypatch.setattr(
            "aiperf.records.record_processor_service.os._exit",
            lambda code: exits.append(code),
        )
        await rp._on_tokenizer_ready(
            GroupTokenizerReady(
                service_id="wgm-0",
                success=False,
                bundles={},
                error_message="404 from hub",
            )
        )
        assert exits == [1]
        rp.error.assert_called()


class TestParserPrefersTheBundle:
    def test_bundle_path_wins_over_the_model_name(self):
        from aiperf.records.inference_result_parser import InferenceResultParser

        parser = InferenceResultParser.__new__(InferenceResultParser)
        parser._tokenizer_bundles = {"my-model": "/pod/tok/my-model"}
        assert parser._tokenizer_source("my-model", "my-model") == "/pod/tok/my-model"

    def test_falls_back_to_the_configured_name(self):
        from aiperf.records.inference_result_parser import InferenceResultParser

        parser = InferenceResultParser.__new__(InferenceResultParser)
        parser._tokenizer_bundles = {}
        assert parser._tokenizer_source("my-model", "hf/my-model") == "hf/my-model"

    def test_explicit_tokenizer_name_finds_the_bundle(self):
        """The WGM keys bundles by tokenizer name, not model name.

        ``_unique_tokenizer_names`` prefers ``cfg.tokenizer.name``, so with an
        explicit tokenizer the two names differ and a model-name lookup misses
        every bundle -- silently reinstating the per-processor hub download
        this feature exists to avoid.
        """
        from aiperf.records.inference_result_parser import InferenceResultParser

        parser = InferenceResultParser.__new__(InferenceResultParser)
        parser._tokenizer_bundles = {"hf/shared-tok": "/pod/tok/shared"}
        assert (
            parser._tokenizer_source("my-model", "hf/shared-tok") == "/pod/tok/shared"
        )

    def test_model_keyed_bundle_still_found_after_alias_resolution(self):
        """With no explicit tokenizer the WGM keys by model name, but CLI alias
        resolution can still hand the parser a different ``configured_name``.
        """
        from aiperf.records.inference_result_parser import InferenceResultParser

        parser = InferenceResultParser.__new__(InferenceResultParser)
        parser._tokenizer_bundles = {"my-model": "/pod/tok/my-model"}
        assert (
            parser._tokenizer_source("my-model", "resolved/my-model")
            == "/pod/tok/my-model"
        )
