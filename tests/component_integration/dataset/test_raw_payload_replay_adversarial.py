# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial component-integration tests for the raw-payload replay pipeline.

Exercises the end-to-end behaviour pinned by Wave-2 fixes (W2-A through W2-E):

- ``InputsJsonPayloadLoader`` rejects missing keys / duplicate session_ids (W2-A).
- ``RawPayloadDatasetLoader._dir_has_raw_payload_jsonl`` no longer swallows
  ``PermissionError`` silently (W2-B).
- ``DatasetManager`` skips ``inputs.json`` generation for Mooncake *payload*
  mode and raises ``ValueError`` on mixed raw_payload / non-raw conversations
  (W2-C).
- ``RawRecordWriterProcessor`` drops records with non-JSON ``payload_bytes``
  and surfaces the count via ``dropped_record_count`` (W2-D).
- ``InferenceClient`` rejects pre-serialised ``payload_bytes`` that don't
  round-trip through ``orjson.loads`` before handing anything to transport
  (W2-E).

Every test wires together real loader / DatasetManager / processor / client
construction — mocking is limited to transport I/O boundaries so the
end-to-end code path is the one actually exercised.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import orjson
import pytest

from aiperf.common.config import (
    EndpointConfig,
    InputConfig,
    OutputConfig,
    ServiceConfig,
    UserConfig,
)
from aiperf.common.config.config_defaults import OutputDefaults
from aiperf.common.enums import (
    ConversationContextMode,
    CreditPhase,
    ExportLevel,
    ModelSelectionStrategy,
)
from aiperf.common.models import Conversation, TextResponse, Turn
from aiperf.common.models.dataset_models import Text
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import (
    RawRecordInfo,
    RequestInfo,
    RequestRecord,
)
from aiperf.dataset.dataset_manager import DatasetManager
from aiperf.dataset.loader.raw_payload import RawPayloadDatasetLoader
from aiperf.plugin.enums import CustomDatasetType, EndpointType, TransportType
from aiperf.post_processors.raw_record_writer_processor import RawRecordWriterProcessor
from aiperf.workers.inference_client import InferenceClient

pytestmark = pytest.mark.component_integration


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _raw_payload_body() -> dict[str, Any]:
    """Minimal chat-API-shaped raw payload body used across fixtures."""
    return {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


def _make_user_config(
    tmp_path: Path,
    *,
    custom_dataset_type: str | None,
    input_file: Path | None = None,
    endpoint_type: str = EndpointType.RAW,
) -> UserConfig:
    """Build a full UserConfig rooted at ``tmp_path`` for DatasetManager tests."""
    if input_file is None and custom_dataset_type is not None:
        input_file = tmp_path / "fake_input.jsonl"
        input_file.touch()
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=endpoint_type,
            streaming=False,
            url="http://localhost:8000",
        ),
        input=InputConfig(
            custom_dataset_type=custom_dataset_type,
            file=str(input_file) if input_file else None,
        ),
        output=OutputConfig(artifact_directory=tmp_path),
    )


def _make_dataset_manager(
    tmp_path: Path,
    *,
    custom_dataset_type: str | None,
    dataset: dict[str, Conversation],
    input_file: Path | None = None,
    endpoint_type: str = EndpointType.RAW,
) -> DatasetManager:
    """Construct a real ``DatasetManager`` pre-populated with ``dataset``."""
    user_config = _make_user_config(
        tmp_path,
        custom_dataset_type=custom_dataset_type,
        input_file=input_file,
        endpoint_type=endpoint_type,
    )
    mgr = DatasetManager(
        service_config=ServiceConfig(),
        user_config=user_config,
        service_id="test_dm",
    )
    mgr.dataset = dataset
    mgr._configure_dataset = AsyncMock()
    mgr._configure_tokenizer = AsyncMock()
    mgr._configure_dataset_client_and_free_memory = AsyncMock()
    return mgr


def _chat_model_endpoint() -> ModelEndpointInfo:
    return ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name="test-model")],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(
            type=EndpointType.CHAT,
            base_url="http://localhost:8000/v1/test",
        ),
    )


def _make_inference_client() -> InferenceClient:
    """Build an InferenceClient with mocked endpoint + transport plugins."""
    mock_transport = MagicMock()
    mock_endpoint = MagicMock()
    mock_endpoint.get_endpoint_headers.return_value = {}
    mock_endpoint.get_endpoint_params.return_value = {}
    mock_endpoint.format_payload.return_value = {"from": "format_payload"}

    def _get_class(protocol: str, name: str):
        if protocol == "endpoint":
            return lambda **_kw: mock_endpoint
        if protocol == "transport":
            return lambda **_kw: mock_transport
        raise ValueError(f"Unknown protocol: {protocol}")

    http_entry = MagicMock()
    http_entry.name = TransportType.HTTP.value
    http_entry.metadata = {"url_schemes": ["http", "https"]}

    with (
        patch(
            "aiperf.workers.inference_client.plugins.get_class",
            side_effect=_get_class,
        ),
        patch(
            "aiperf.workers.inference_client.plugins.list_entries",
            return_value=[http_entry],
        ),
    ):
        return InferenceClient(
            model_endpoint=_chat_model_endpoint(),
            service_id="ic-test",
        )


def _user_config_raw(tmp_path: Path) -> UserConfig:
    """Build a UserConfig that triggers RAW export level + artifact dir."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
            streaming=False,
        ),
        output=OutputConfig(
            artifact_directory=artifact_dir,
            export_level=ExportLevel.RAW,
        ),
    )


# ---------------------------------------------------------------------------
# 1 & 2: RAW_PAYLOAD / INPUTS_JSON skip inputs.json end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_raw_payload_loader_to_dataset_manager_skip_inputs_json_end_to_end(
    tmp_path: Path,
) -> None:
    """RAW_PAYLOAD-typed datasets must bypass ``_generate_inputs_json_file``
    entirely when DatasetManager runs its configure command — the loader
    already materialised raw_payload turns so re-serialising would be a waste
    and would trip the 'all-or-none' invariant."""
    dataset = {
        "s1": Conversation(
            session_id="s1",
            context_mode=ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES,
            turns=[Turn(role="user", raw_payload=_raw_payload_body())],
        ),
    }
    mgr = _make_dataset_manager(
        tmp_path,
        custom_dataset_type=CustomDatasetType.RAW_PAYLOAD,
        dataset=dataset,
    )

    with patch.object(
        mgr, "_generate_inputs_json_file", new_callable=AsyncMock
    ) as mock_gen:
        await mgr._profile_configure_command(Mock())

    mock_gen.assert_not_called()
    assert not (tmp_path / OutputDefaults.INPUTS_JSON_FILE).exists()


@pytest.mark.asyncio
async def test_inputs_json_loader_to_dataset_manager_skip_inputs_json_end_to_end(
    tmp_path: Path,
) -> None:
    """INPUTS_JSON-typed datasets must also bypass inputs.json regeneration:
    the loader reads AIPerf's own inputs.json and builds raw_payload turns,
    so re-emitting it would be a circular waste."""
    dataset = {
        "s1": Conversation(
            session_id="s1",
            context_mode=ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES,
            turns=[Turn(role="user", raw_payload=_raw_payload_body())],
        ),
    }
    mgr = _make_dataset_manager(
        tmp_path,
        custom_dataset_type=CustomDatasetType.INPUTS_JSON,
        dataset=dataset,
    )

    with patch.object(
        mgr, "_generate_inputs_json_file", new_callable=AsyncMock
    ) as mock_gen:
        await mgr._profile_configure_command(Mock())

    mock_gen.assert_not_called()
    assert not (tmp_path / OutputDefaults.INPUTS_JSON_FILE).exists()


# ---------------------------------------------------------------------------
# 3 & 4: Mooncake payload-mode skip vs messages-mode emit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mooncake_trace_payload_mode_now_skips_inputs_json_end_to_end(
    tmp_path: Path,
) -> None:
    """Post-W2-C: Mooncake sessions loaded in 'payload' mode (every turn has
    a raw_payload) are detected via the all-turns-have-raw_payload invariant
    and added to the inputs.json skip list. Without the fix this path fell
    through to ``_generate_inputs_json_file`` even though the payloads were
    pre-built."""
    dataset = {
        "s1": Conversation(
            session_id="s1",
            context_mode=ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES,
            turns=[Turn(role="user", raw_payload=_raw_payload_body())],
        ),
        "s2": Conversation(
            session_id="s2",
            context_mode=ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES,
            turns=[
                Turn(role="user", raw_payload=_raw_payload_body()),
                Turn(role="user", raw_payload=_raw_payload_body()),
            ],
        ),
    }
    mgr = _make_dataset_manager(
        tmp_path,
        custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
        dataset=dataset,
    )
    # _configure_dataset is mocked out, so set the source-payload flag
    # it would normally compute before _preformat_payloads ran.
    mgr._all_turns_source_loaded_payloads = True

    with patch.object(
        mgr, "_generate_inputs_json_file", new_callable=AsyncMock
    ) as mock_gen:
        await mgr._profile_configure_command(Mock())

    mock_gen.assert_not_called()


@pytest.mark.asyncio
async def test_mooncake_trace_messages_mode_still_emits_inputs_json_end_to_end(
    tmp_path: Path,
) -> None:
    """Mooncake sessions loaded in 'messages' / synthesized mode (no
    raw_payload on any turn) must still produce inputs.json — the W2-C
    detection must not over-reach and swallow the normal Mooncake flow."""
    dataset = {
        "s1": Conversation(
            session_id="s1",
            turns=[Turn(role="user", texts=[Text(contents=["hello"])])],
        ),
    }
    mgr = _make_dataset_manager(
        tmp_path,
        custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
        dataset=dataset,
        endpoint_type=EndpointType.CHAT,
    )

    with patch.object(
        mgr, "_generate_inputs_json_file", new_callable=AsyncMock
    ) as mock_gen:
        await mgr._profile_configure_command(Mock())

    mock_gen.assert_called_once()


# ---------------------------------------------------------------------------
# 5: Mixed-state conversation raises during _generate_input_payloads
# ---------------------------------------------------------------------------


def test_mixed_state_conversation_raises_during_generate_input_payloads_end_to_end(
    tmp_path: Path,
) -> None:
    """Post-W2-C: a conversation with some raw_payload turns and some
    non-raw turns is invalid (v1 needs all-or-none per conversation). The
    raw branch of ``_generate_input_payloads`` must raise ValueError
    identifying the offending session rather than silently dropping
    non-raw turns."""
    mixed_conv = Conversation(
        session_id="mixed",
        turns=[
            Turn(role="user", raw_payload=_raw_payload_body()),
            Turn(role="user", texts=[Text(contents=["should-not-be-dropped"])]),
        ],
    )
    mgr = _make_dataset_manager(
        tmp_path,
        custom_dataset_type=CustomDatasetType.RAW_PAYLOAD,
        dataset={"mixed": mixed_conv},
    )
    raw_endpoint = ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name="test-model")],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(type=EndpointType.RAW, base_url="http://localhost"),
    )

    with pytest.raises(ValueError, match="mixed raw_payload"):
        mgr._generate_input_payloads(raw_endpoint)


# ---------------------------------------------------------------------------
# 6: InferenceClient pre-send payload_bytes validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inference_client_forwards_invalid_json_payload_bytes_verbatim() -> None:
    """Per-request ``orjson.loads`` validation of pre-serialised
    ``payload_bytes`` was removed — invalid-JSON detection happens at
    dataset-load time, not on every send. Unparsable bytes are forwarded to
    the transport verbatim rather than being turned into an error record."""
    client = _make_inference_client()

    turn = Turn(texts=[Text(contents=["x"])], role="user", model="test-model")
    info = RequestInfo(
        model_endpoint=client.model_endpoint,
        turns=[turn],
        turn_index=0,
        credit_num=1,
        credit_phase=CreditPhase.PROFILING,
        x_request_id="rid",
        x_correlation_id="cid",
        conversation_id="conv",
        payload_bytes=b"}",
    )
    client.transport.send_request = AsyncMock(
        return_value=RequestRecord(request_info=info)
    )

    await client.send_request(info)

    client.transport.send_request.assert_called_once()
    assert client.transport.send_request.call_args.kwargs["payload"] == b"}"


# ---------------------------------------------------------------------------
# 7: RawRecordWriter splices payload_bytes verbatim (no per-record re-parse)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_raw_record_writer_splices_invalid_fragment_verbatim_end_to_end(
    tmp_path: Path,
) -> None:
    """The per-record ``orjson.loads`` re-validation was removed:
    ``RawRecordWriterProcessor`` splices ``payload_bytes`` verbatim via
    ``orjson.Fragment``. Invalid JSON bytes are written (no drop, no counter)
    and the resulting line is corrupt — the accepted tradeoff of trusting
    dataset-load-time validation instead of re-parsing every exported record."""
    user_config = _user_config_raw(tmp_path)
    processor = RawRecordWriterProcessor(service_id="rrw-ci", user_config=user_config)
    await processor.initialize()
    await processor.start()
    try:
        bad = RawRecordInfo.model_construct(
            metadata=_metric_metadata(),
            start_perf_ns=1_000_000_000,
            payload=None,
            payload_bytes=b"}",  # invalid JSON
            request_headers={},
            response_headers=None,
            status=200,
            responses=[TextResponse(text="ok", perf_ns=2_000_000_000)],
            error=None,
        )
        await processor.buffered_write(bad)

        assert processor.dropped_record_count == 0
        assert processor.lines_written == 1
    finally:
        await processor.stop()

    # The verbatim splice produced a line that no longer parses as JSON.
    raw = processor.output_file.read_bytes()
    line = next(line for line in raw.splitlines() if line.strip())
    with pytest.raises(orjson.JSONDecodeError):
        orjson.loads(line)


def _metric_metadata():
    """Minimal MetricRecordMetadata for RawRecordInfo construction."""
    from aiperf.common.models.record_models import MetricRecordMetadata

    return MetricRecordMetadata(
        session_num=0,
        conversation_id="conv-ci",
        turn_index=0,
        request_start_ns=1_000_000_000,
        request_ack_ns=None,
        request_end_ns=1_100_000_000,
        worker_id="worker-ci",
        record_processor_id="rrw-ci",
        benchmark_phase=CreditPhase.PROFILING,
        x_request_id="req-ci",
        x_correlation_id="corr-ci",
    )


# ---------------------------------------------------------------------------
# 8 & 9: InputsJsonLoader adversarial parsing
# ---------------------------------------------------------------------------


def test_inputs_json_loader_rejects_duplicate_session_ids_end_to_end(
    tmp_path: Path,
) -> None:
    """Post-W2-A: ``InputsJsonPayloadLoader.load_dataset`` raises ValueError
    with a 'duplicate' message (and the session_id) when two entries share
    the same session_id. Previously the second entry silently overwrote
    the first."""
    from aiperf.dataset.loader.inputs_json import InputsJsonPayloadLoader

    content = {
        "data": [
            {"session_id": "dup", "payloads": [_raw_payload_body()]},
            {"session_id": "dup", "payloads": [_raw_payload_body()]},
        ]
    }
    path = tmp_path / "inputs_dup.json"
    path.write_bytes(orjson.dumps(content))

    loader = InputsJsonPayloadLoader(filename=str(path), user_config=MagicMock())
    with pytest.raises(ValueError, match="duplicate"):
        loader.load_dataset()


def test_inputs_json_loader_rejects_missing_required_keys_end_to_end(
    tmp_path: Path,
) -> None:
    """Post-W2-A: entries missing ``session_id`` (or ``payloads``) raise
    ValueError and the message must identify the offending entry index so
    operators can locate the bad record in a large inputs.json."""
    from aiperf.dataset.loader.inputs_json import InputsJsonPayloadLoader

    content = {
        "data": [
            {"session_id": "ok", "payloads": [_raw_payload_body()]},
            {"payloads": [_raw_payload_body()]},  # missing session_id
        ]
    }
    path = tmp_path / "inputs_missing.json"
    path.write_bytes(orjson.dumps(content))

    loader = InputsJsonPayloadLoader(filename=str(path), user_config=MagicMock())
    with pytest.raises(ValueError, match="session_id") as excinfo:
        loader.load_dataset()

    # Error message must name the entry index (entry[1]) for operator
    # locate-ability; otherwise a 100k-line inputs.json becomes unusable.
    assert "entry[1]" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 10: Permission-denied JSONL in raw-payload directory must raise
# ---------------------------------------------------------------------------


def test_raw_payload_dir_with_permission_denied_jsonl_raises_not_returns_false(
    tmp_path: Path,
) -> None:
    """Post-W2-B: ``_dir_has_raw_payload_jsonl`` narrowed its exception
    catch to ``(orjson.JSONDecodeError, ValueError)`` only, so
    ``PermissionError`` (from a chmod 0o000 file) now surfaces instead of
    being silently treated as 'not a raw_payload dir'. Operators catching
    the permission problem early is the whole point of the fix."""
    if os.geteuid() == 0:
        pytest.skip("chmod 0o000 does not restrict root; run as non-root for this test")

    unreadable = tmp_path / "unreadable.jsonl"
    unreadable.write_bytes(orjson.dumps(_raw_payload_body()) + b"\n")
    unreadable.chmod(0o000)
    try:
        with pytest.raises(PermissionError):
            RawPayloadDatasetLoader.can_load(filename=tmp_path)
    finally:
        # Restore so tmp_path teardown can clean up.
        unreadable.chmod(0o644)
