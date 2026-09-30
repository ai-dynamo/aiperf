# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import orjson
import pytest
import zstandard
from pytest import param

from aiperf.common.exceptions import DatasetLoaderError
from aiperf.common.tokenizer import Tokenizer
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.dataset.composer.custom import CustomDatasetComposer
from aiperf.dataset.loader.h_cua_perf_file import HCuaPerfFileLoader
from aiperf.plugin import plugins
from aiperf.plugin.enums import CustomDatasetType, PluginType
from tests.unit.conftest import make_run_from_cli
from tests.unit.dataset.loader._h_cua_perf_records import trajectory

SESSION_TURNS = {"traj-a": 3, "traj-b": 1, "traj-c": 2}
RECORDS = [r for sid, n in SESSION_TURNS.items() for r in trajectory(sid, n)]


def build(directory: Path, name: str) -> Path:
    """A build as trace_processor.py writes it: the trace, plain or zstd, beside its manifest."""
    trace = directory / name
    lines = b"".join(orjson.dumps(r) + b"\n" for r in RECORDS)
    trace.write_bytes(zstandard.compress(lines) if name.endswith(".zst") else lines)
    digest = hashlib.sha256(trace.read_bytes()).hexdigest()
    (directory / "h_cua.meta.json").write_bytes(
        orjson.dumps({"session_turns": SESSION_TURNS, "sha256": {name: digest}})
    )
    return trace


def _run(trace: Path) -> BenchmarkRun:
    return make_run_from_cli(
        CLIConfig(
            model_names=["test-model"],
            endpoint_type="chat",
            input_file=str(trace),
            custom_dataset_type=CustomDatasetType.H_CUA_PERF,
        )
    )


class TestRegistry:
    def test_plugin_resolves_to_loader_class(self) -> None:
        cls = plugins.get_class(
            PluginType.CUSTOM_DATASET_LOADER, CustomDatasetType.H_CUA_PERF
        )
        assert cls is HCuaPerfFileLoader

    def test_never_claims_a_file(self, tmp_path: Path) -> None:
        """Its rows are Mooncake rows; claiming them would break mooncake_trace detection."""
        assert (
            HCuaPerfFileLoader.can_load(RECORDS[0], tmp_path / "h_cua.jsonl") is False
        )


class TestLoader:
    """The Hub loader's tests cover the replay itself; these cover the file path into it."""

    @pytest.mark.parametrize(
        "name",
        [
            param("h_cua.jsonl.zst", id="zstd"),
            param("h_cua.jsonl", id="plain"),
        ],
    )  # fmt: skip
    def test_composer_replays_the_build_named_by_input_file(
        self, tmp_path: Path, name: str, mock_tokenizer_cls: type[Tokenizer]
    ) -> None:
        trace = build(tmp_path, name)
        composer = CustomDatasetComposer(
            run=_run(trace), tokenizer=mock_tokenizer_cls.from_pretrained("test-model")
        )

        conversations = composer.create_dataset()

        assert isinstance(composer.loader, HCuaPerfFileLoader)
        assert {c.session_id: len(c.turns) for c in conversations} == SESSION_TURNS

    def test_missing_manifest_is_rejected_at_construction(self, tmp_path: Path) -> None:
        trace = build(tmp_path, "h_cua.jsonl.zst")
        (tmp_path / "h_cua.meta.json").unlink()

        with pytest.raises(DatasetLoaderError, match="h_cua.meta.json"):
            HCuaPerfFileLoader(
                filename=trace, prompt_generator=MagicMock(), run=_run(trace)
            )
