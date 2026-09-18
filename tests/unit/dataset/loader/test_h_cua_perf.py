# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import orjson
import pytest
import zstandard

from aiperf.common.enums import ConversationContextMode
from aiperf.common.exceptions import DatasetLoaderError
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.loader.h_cua_perf import HCuaPerfDatasetLoader
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, PublicDatasetType
from tests.unit.conftest import make_run_from_cli
from tests.unit.dataset.loader._h_cua_perf_records import (
    TOOLS,
    image_slots,
    trajectory,
)

HF_DATASET_NAME = "Hcompany/h_cua_perf"
SESSION_TURNS = {"traj-a": 3, "traj-b": 1, "traj-c": 2}
RECORDS = [r for sid, n in SESSION_TURNS.items() for r in trajectory(sid, n)]


@pytest.fixture
def hub(tmp_path: Path):
    """Serve the fixture manifest and trace through a patched ``hf_hub_download``."""
    files = {
        "h_cua.meta.json": tmp_path / "h_cua.meta.json",
        "h_cua.jsonl.zst": tmp_path / "h_cua.jsonl.zst",
    }
    files["h_cua.meta.json"].write_bytes(orjson.dumps({"session_turns": SESSION_TURNS}))
    with zstandard.open(files["h_cua.jsonl.zst"], "wb") as f:
        for r in RECORDS:
            f.write(orjson.dumps(r) + b"\n")
    with patch(
        "huggingface_hub.hf_hub_download",
        side_effect=lambda repo_id, filename, repo_type, revision: str(files[filename]),
    ) as download:
        yield download


def _loader(filters: dict[str, str] | None = None, **cli: Any) -> HCuaPerfDatasetLoader:
    run = make_run_from_cli(
        CLIConfig(
            model_names=["test-model"],
            endpoint_type="chat",
            public_dataset=PublicDatasetType.H_CUA_PERF,
            **cli,
        )
    )
    return HCuaPerfDatasetLoader(
        run=run,
        hf_dataset_name=HF_DATASET_NAME,
        prompt_generator=MagicMock(),
        filters=filters,
    )


class TestRegistry:
    def test_plugin_resolves_to_loader_class(self) -> None:
        cls = plugins.get_class(
            PluginType.PUBLIC_DATASET_LOADER, PublicDatasetType.H_CUA_PERF
        )
        assert cls is HCuaPerfDatasetLoader

    def test_metadata_declares_trace_with_timing(self) -> None:
        metadata = plugins.get_public_dataset_loader_metadata(
            PublicDatasetType.H_CUA_PERF
        )
        assert metadata.hf_dataset_name == HF_DATASET_NAME
        assert (metadata.is_trace, metadata.has_timing_data) == (True, False)


@pytest.mark.asyncio
class TestLoader:
    """Async: the base class opens an aiohttp client at construction."""

    async def test_invalid_filters_are_rejected(self) -> None:
        with pytest.raises(DatasetLoaderError, match="supported keys: n_screenshots"):
            _loader({"screenshots": "3"})

    async def test_end_to_end_replays_selected_trajectories_verbatim(
        self, hub: MagicMock
    ) -> None:
        loader = _loader(
            {"max_trace_length": "2", "n_screenshots": "2"},
            conversation_num_dataset_entries=2,
        )

        data = await loader.load_dataset()
        conversations = await loader.convert_to_conversations(data)

        assert [c.kwargs["filename"] for c in hub.call_args_list] == [
            "h_cua.meta.json",
            "h_cua.jsonl.zst",
        ]
        assert {c.kwargs["repo_id"] for c in hub.call_args_list} == {HF_DATASET_NAME}
        assert {sid: len(traces) for sid, traces in data.items()} == {
            "traj-a": 2,
            "traj-b": 1,
        }

        by_id = {c.session_id: c for c in conversations}
        assert (
            by_id["traj-a"].context_mode
            == ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
        )
        first, second = by_id["traj-a"].turns
        assert first.raw_messages == RECORDS[0]["messages"]
        assert first.raw_tools == TOOLS
        assert first.extra_body == {"tool_choice": "auto"}
        assert (first.max_tokens, first.timestamp, first.delay) == (10, None, None)
        assert (second.max_tokens, second.timestamp, second.delay) == (11, None, 2000)
        assert image_slots(second.raw_messages) == [0, 1]

    async def test_request_count_fallback_does_not_cap_trajectories(
        self, hub: MagicMock
    ) -> None:
        data = await _loader(request_count=1).load_dataset()
        assert {sid: len(t) for sid, t in data.items()} == SESSION_TURNS

    async def test_manifest_without_session_turns_is_rejected(
        self, hub: MagicMock, tmp_path: Path
    ) -> None:
        (tmp_path / "h_cua.meta.json").write_bytes(orjson.dumps({"num_traces": 3}))
        with pytest.raises(DatasetLoaderError, match="session_turns"):
            await _loader().load_dataset()
