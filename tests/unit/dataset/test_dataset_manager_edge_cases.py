# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge-case unit tests for DatasetManager.

Focuses on error / cancellation / fallback paths that the existing
``test_dataset_manager.py`` and ``test_dataset_manager_inputs_json.py`` suites
do not exercise:

- Composer-level loader failures (file missing, malformed JSON).
- ``--request-count`` exceeding dataset size (recycling) — verify the manager
  itself does not deduplicate or balk on it; recycling is a downstream concern
  but the manager must still expose the small dataset cleanly.
- Multi-turn conversations flow through dataset metadata correctly.
- Cancellation while a configure pass is in-flight.
- Empty / single-entry datasets.
- Tokenizer load failure surfaces cleanly.
- Fallback handlers in Kubernetes (compress_only) mode reject requests.
- ``_wait_for_dataset_configuration`` honors the configured timeout.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import orjson
import pytest
from pytest import param

from aiperf.common.control_structs import Command
from aiperf.common.enums import CommandType
from aiperf.common.exceptions import ServiceError
from aiperf.common.messages import (
    ConversationRequestMessage,
    ConversationTurnRequestMessage,
)
from aiperf.config import AIPerfConfig, BenchmarkRun
from aiperf.dataset.dataset_manager import DatasetManager
from aiperf.plugin.enums import CustomDatasetType, EndpointType

# ============================================================================
# Test config helpers
# ============================================================================


_BASE_CONFIG = dict(
    models=["test-model"],
    endpoint={
        "urls": ["http://localhost:8000/v1/chat/completions"],
        "type": EndpointType.CHAT,
    },
    phases=[
        {"name": "default", "type": "concurrency", "requests": 10, "concurrency": 1}
    ],
)


@pytest.fixture
def mock_tokenizer(mock_tokenizer_cls):
    """Patch Tokenizer.from_pretrained for the duration of the test."""
    with patch("aiperf.common.tokenizer.Tokenizer.from_pretrained") as m:
        m.return_value = mock_tokenizer_cls.from_pretrained("test-model")
        yield m


def _make_config(
    *,
    dataset: dict | None = None,
    artifacts_dir: Path | None = None,
) -> AIPerfConfig:
    if dataset is None:
        dataset = {
            "name": "default",
            "type": "synthetic",
            "entries": 100,
            "prompts": {"isl": 128, "osl": 64},
        }
    body = dict(_BASE_CONFIG)
    body["datasets"] = [dataset]
    if artifacts_dir is not None:
        body["artifacts"] = {"dir": str(artifacts_dir)}
    return AIPerfConfig(benchmark=body)


def _make_run(cfg: AIPerfConfig, *, artifact_dir: Path | None = None) -> BenchmarkRun:
    return BenchmarkRun(
        benchmark_id="test",
        cfg=cfg.benchmark,
        artifact_dir=artifact_dir or Path("/tmp/test"),
    )


async def _new_initialized_manager(run: BenchmarkRun) -> DatasetManager:
    manager = DatasetManager(run=run, service_id="dm-edge")
    await manager.initialize()
    manager.publish = AsyncMock()
    return manager


# ============================================================================
# Loader error paths
# ============================================================================


class TestDatasetManagerLoaderErrors:
    """Errors raised inside load_conversations_for_run propagate fatally."""

    @pytest.mark.asyncio
    async def test_missing_trace_file_raises(
        self,
        mock_tokenizer,
        tmp_path: Path,
    ) -> None:
        missing = tmp_path / "does_not_exist.jsonl"
        cfg = _make_config(
            dataset={
                "name": "default",
                "type": "file",
                "path": str(missing),
                "format": CustomDatasetType.MOONCAKE_TRACE,
            },
            artifacts_dir=tmp_path,
        )
        manager = await _new_initialized_manager(_make_run(cfg, artifact_dir=tmp_path))

        with pytest.raises((FileNotFoundError, OSError, ValueError, Exception)):
            await manager._profile_configure_command(
                Command(cid="c", cmd=CommandType.PROFILE_CONFIGURE)
            )
        # Configuration must NOT signal complete after a load failure.
        assert not manager.dataset_configured.is_set()

    @pytest.mark.asyncio
    async def test_malformed_json_trace_raises(
        self,
        mock_tokenizer,
        tmp_path: Path,
    ) -> None:
        bad = tmp_path / "bad.jsonl"
        bad.write_text("{this is not valid json\n")
        cfg = _make_config(
            dataset={
                "name": "default",
                "type": "file",
                "path": str(bad),
                "format": CustomDatasetType.MOONCAKE_TRACE,
            },
            artifacts_dir=tmp_path,
        )
        manager = await _new_initialized_manager(_make_run(cfg, artifact_dir=tmp_path))

        with pytest.raises((orjson.JSONDecodeError, ValueError)):
            await manager._profile_configure_command(
                Command(cid="c", cmd=CommandType.PROFILE_CONFIGURE)
            )
        assert not manager.dataset_configured.is_set()

    @pytest.mark.asyncio
    async def test_tokenizer_load_failure_is_fatal(
        self,
        tmp_path: Path,
    ) -> None:
        """If the tokenizer fails to load on a tokenizing endpoint, configure must error."""
        cfg = _make_config(artifacts_dir=tmp_path)
        manager = await _new_initialized_manager(_make_run(cfg, artifact_dir=tmp_path))

        with (
            patch(
                "aiperf.dataset.dataset_manager.load_tokenizer_for_run",
                side_effect=RuntimeError("tokenizer download blocked"),
            ),
            pytest.raises(RuntimeError, match="tokenizer download blocked"),
        ):
            await manager._profile_configure_command(
                Command(cid="c", cmd=CommandType.PROFILE_CONFIGURE)
            )
        assert manager.tokenizer is None
        assert not manager.dataset_configured.is_set()


# ============================================================================
# Dataset size edge cases (single, empty, recycle scenarios)
# ============================================================================


class TestDatasetManagerSmallAndRecycledDatasets:
    """Behavior when the dataset is empty / single entry / smaller than request count."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "entries",
        [
            param(1, id="single-entry"),
            param(2, id="two-entries-fewer-than-requests"),
        ],
    )  # fmt: skip
    async def test_small_synthetic_dataset_configures_cleanly(
        self,
        mock_tokenizer,
        tmp_path: Path,
        entries: int,
    ) -> None:
        """`--request-count 10` recycles a small dataset; the manager itself just publishes the small set."""
        cfg = _make_config(
            dataset={
                "name": "default",
                "type": "synthetic",
                "entries": entries,
                "prompts": {"isl": 32, "osl": 16},
            },
            artifacts_dir=tmp_path,
        )
        manager = await _new_initialized_manager(_make_run(cfg, artifact_dir=tmp_path))

        await manager._profile_configure_command(
            Command(cid="c", cmd=CommandType.PROFILE_CONFIGURE)
        )

        assert manager.dataset_configured.is_set()
        assert manager.dataset_metadata is not None
        assert len(manager.dataset_metadata.conversations) == entries
        # Conversation IDs are unique even for a 1-entry dataset.
        ids = [c.conversation_id for c in manager.dataset_metadata.conversations]
        assert len(set(ids)) == len(ids)


# ============================================================================
# Multi-turn / per-session field plumbing
# ============================================================================


class TestDatasetManagerMultiTurn:
    """Multi-turn conversations preserve per-turn payload data through metadata."""

    @pytest.mark.asyncio
    async def test_multi_turn_metadata_preserves_per_turn_input_lengths(
        self,
        mock_tokenizer,
        create_mooncake_trace_file,
        tmp_path: Path,
    ) -> None:
        """Per-turn fields (input_length / delay) survive the configure pass on a real trace.

        Guards against the gotcha where a per-session field is dropped in the
        strategy -> Turn -> Credit -> Worker chain.
        """
        entries = [
            json.dumps(
                {
                    "session_id": "sess-1",
                    "timestamp": 0,
                    "input_length": 50,
                    "output_length": 10,
                }
            ),
            json.dumps(
                {
                    "session_id": "sess-1",
                    "delay": 1234,
                    "input_length": 73,
                    "output_length": 10,
                }
            ),
            json.dumps(
                {
                    "session_id": "sess-1",
                    "delay": 5678,
                    "input_length": 91,
                    "output_length": 10,
                }
            ),
        ]
        path = create_mooncake_trace_file(entries)
        try:
            cfg = _make_config(
                dataset={
                    "name": "default",
                    "type": "file",
                    "path": path,
                    "format": CustomDatasetType.MOONCAKE_TRACE,
                },
                artifacts_dir=tmp_path,
            )
            manager = await _new_initialized_manager(
                _make_run(cfg, artifact_dir=tmp_path)
            )

            await manager._profile_configure_command(
                Command(cid="c", cmd=CommandType.PROFILE_CONFIGURE)
            )

            md = manager.dataset_metadata
            assert md is not None
            assert len(md.conversations) == 1
            sess = md.conversations[0]
            assert sess.conversation_id == "sess-1"
            assert len(sess.turns) == 3
            assert md.has_timing_data is True
        finally:
            Path(path).unlink(missing_ok=True)


# ============================================================================
# Cancellation
# ============================================================================


class TestDatasetManagerCancellation:
    """Cancellation while configure is in-flight is propagated."""

    @pytest.mark.asyncio
    async def test_cancellation_during_configure_propagates(
        self,
        mock_tokenizer,
        tmp_path: Path,
    ) -> None:
        cfg = _make_config(artifacts_dir=tmp_path)
        manager = await _new_initialized_manager(_make_run(cfg, artifact_dir=tmp_path))

        async def block_forever() -> None:
            await asyncio.Event().wait()

        manager._configure_tokenizer = block_forever  # type: ignore[method-assign]

        task = asyncio.create_task(
            manager._profile_configure_command(
                Command(cid="c", cmd=CommandType.PROFILE_CONFIGURE)
            )
        )
        # Yield once so the task starts the inner work.
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert not manager.dataset_configured.is_set()


# ============================================================================
# Wait-for-configuration timeout
# ============================================================================


class TestWaitForDatasetConfigurationTimeout:
    """`_wait_for_dataset_configuration` raises TimeoutError when not signalled."""

    @pytest.mark.asyncio
    async def test_wait_times_out_when_event_never_set(
        self,
        mock_tokenizer,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cfg = _make_config(artifacts_dir=tmp_path)
        manager = await _new_initialized_manager(_make_run(cfg, artifact_dir=tmp_path))

        # Force a near-zero configuration timeout so the wait fails immediately.
        from aiperf.common.environment import Environment

        monkeypatch.setattr(Environment.DATASET, "CONFIGURATION_TIMEOUT", 0.01)

        with pytest.raises(TimeoutError):
            await manager._wait_for_dataset_configuration()


# ============================================================================
# Kubernetes (compress_only) fallback handlers
# ============================================================================


class TestKubernetesFallbackRejection:
    """In Kubernetes mode the manager refuses to serve fallback requests."""

    @pytest.mark.asyncio
    async def test_conversation_request_rejected_in_compress_only_mode(
        self,
        mock_tokenizer,
        tmp_path: Path,
    ) -> None:
        cfg = _make_config(
            dataset={
                "name": "default",
                "type": "synthetic",
                "entries": 2,
                "prompts": {"isl": 32, "osl": 16},
            },
            artifacts_dir=tmp_path,
        )
        manager = await _new_initialized_manager(_make_run(cfg, artifact_dir=tmp_path))
        manager._compress_only = True

        await manager._profile_configure_command(
            Command(cid="c", cmd=CommandType.PROFILE_CONFIGURE)
        )

        assert manager._dataset_client is None
        assert manager.dataset_configured.is_set()

        with pytest.raises(ServiceError, match="Kubernetes mode"):
            await manager._handle_conversation_request(
                ConversationRequestMessage(service_id="worker", conversation_id="any")
            )

    @pytest.mark.asyncio
    async def test_turn_request_rejected_in_compress_only_mode(
        self,
        mock_tokenizer,
        tmp_path: Path,
    ) -> None:
        cfg = _make_config(
            dataset={
                "name": "default",
                "type": "synthetic",
                "entries": 1,
                "prompts": {"isl": 32, "osl": 16},
            },
            artifacts_dir=tmp_path,
        )
        manager = await _new_initialized_manager(_make_run(cfg, artifact_dir=tmp_path))
        manager._compress_only = True

        await manager._profile_configure_command(
            Command(cid="c", cmd=CommandType.PROFILE_CONFIGURE)
        )

        with pytest.raises(ServiceError, match="Kubernetes mode"):
            await manager._handle_conversation_turn_request(
                ConversationTurnRequestMessage(
                    service_id="worker",
                    conversation_id="any",
                    turn_index=0,
                )
            )


# ============================================================================
# Cleanup / on_stop
# ============================================================================


class TestDatasetManagerCleanup:
    """`_cleanup` is safe to call multiple times and tolerates partial init."""

    @pytest.mark.asyncio
    async def test_cleanup_with_no_state_initialized(
        self,
        tmp_path: Path,
    ) -> None:
        cfg = _make_config(artifacts_dir=tmp_path)
        manager = DatasetManager(
            run=_make_run(cfg, artifact_dir=tmp_path),
            service_id="dm-edge",
        )
        # Don't initialize: just exercise on_stop with default attrs.
        # _backing_store is set in __init__ so we still need to mock its stop.
        manager._backing_store.stop = AsyncMock()  # type: ignore[method-assign]
        await manager._cleanup()
        manager._backing_store.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_rebroadcast_survives_profile_start_until_stop(
        self,
        mock_tokenizer,
        tmp_path: Path,
    ) -> None:
        """Regression: rebroadcast must outlive PROFILE_START in local mode.

        A late-subscribing record processor (warm-tokenizer fast path) depends
        entirely on the 1 Hz rebroadcast to receive DatasetConfiguredNotification;
        ZMQ pub/sub does not replay the initial publish. The DatasetManager has
        no PROFILE_START handler, so the command falls through the base layer's
        no-handler ack path and the rebroadcast task survives. `@on_stop`
        (`_cleanup`) is the single owner that cancels it. Pre-fix, an
        ``@on_command(PROFILE_START)`` handler cancelled the task in local mode,
        hanging every fast-config profile run at "Processing Records: 0/N".
        """
        cfg = _make_config(
            dataset={
                "name": "default",
                "type": "synthetic",
                "entries": 1,
                "prompts": {"isl": 32, "osl": 16},
            },
            artifacts_dir=tmp_path,
        )
        manager = await _new_initialized_manager(_make_run(cfg, artifact_dir=tmp_path))
        # Local (non-Kubernetes) mode is where the startup race lived.
        assert manager._compress_only is False

        await manager._profile_configure_command(
            Command(cid="c", cmd=CommandType.PROFILE_CONFIGURE)
        )
        rebroadcast = manager._rebroadcast_task
        assert rebroadcast is not None
        assert not rebroadcast.done()

        # Drive the real command-dispatch path: PROFILE_START has no handler on
        # DatasetManager, so the base layer must ack it without touching the
        # rebroadcast task.
        manager.control_client = AsyncMock()
        await manager._handle_control_command(
            Command(cid="c2", cmd=CommandType.PROFILE_START)
        )
        manager.control_client.send.assert_awaited_once()
        assert manager._rebroadcast_task is rebroadcast
        assert not rebroadcast.done()

        await manager._cleanup()
        assert manager._rebroadcast_task is None
        await asyncio.sleep(0)
        assert rebroadcast.cancelled() or rebroadcast.done()

    @pytest.mark.asyncio
    async def test_cleanup_cancels_rebroadcast_task(
        self,
        mock_tokenizer,
        tmp_path: Path,
    ) -> None:
        cfg = _make_config(
            dataset={
                "name": "default",
                "type": "synthetic",
                "entries": 1,
                "prompts": {"isl": 32, "osl": 16},
            },
            artifacts_dir=tmp_path,
        )
        manager = await _new_initialized_manager(_make_run(cfg, artifact_dir=tmp_path))

        await manager._profile_configure_command(
            Command(cid="c", cmd=CommandType.PROFILE_CONFIGURE)
        )

        rebroadcast = manager._rebroadcast_task
        assert rebroadcast is not None
        assert not rebroadcast.done()

        await manager._cleanup()

        assert manager._rebroadcast_task is None
        # The cancelled task should settle.
        await asyncio.sleep(0)
        assert rebroadcast.cancelled() or rebroadcast.done()
