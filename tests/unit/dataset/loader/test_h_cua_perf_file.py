# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

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
from aiperf.plugin.enums import CustomDatasetType
from tests.unit.conftest import make_run_from_cli
from tests.unit.dataset.loader._h_cua_perf_records import trajectory

SESSION_TURNS = {"traj-a": 3, "traj-b": 1, "traj-c": 2}
RECORDS = [r for sid, n in SESSION_TURNS.items() for r in trajectory(sid, n)]


def build(directory: Path, name: str) -> Path:
    trace = directory / name
    lines = b"".join(orjson.dumps(r) + b"\n" for r in RECORDS)
    trace.write_bytes(zstandard.compress(lines) if name.endswith(".zst") else lines)
    digest = hashlib.sha256(trace.read_bytes()).hexdigest()
    (directory / "h_cua.meta.json").write_bytes(
        orjson.dumps({"session_turns": SESSION_TURNS, "sha256": {name: digest}})
    )
    return trace


def _run(trace: Path, **cli: Any) -> BenchmarkRun:
    return make_run_from_cli(
        CLIConfig(
            model_names=["test-model"],
            endpoint_type="chat",
            input_file=str(trace),
            custom_dataset_type=CustomDatasetType.H_CUA_PERF,
            **cli,
        )
    )


class TestLoader:
    """The Hub loader's tests cover the replay itself; these cover the file path into it."""

    def test_never_claims_a_file(self, tmp_path: Path) -> None:
        """Its rows are Mooncake rows; claiming them would break mooncake_trace detection."""
        assert (
            HCuaPerfFileLoader.can_load(RECORDS[0], tmp_path / "h_cua.jsonl") is False
        )

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

        # The manifest check's digest scopes the hash_ids; the base class must not hash again.
        with patch(
            "aiperf.dataset.loader.base_trace_loader._compute_file_hash"
        ) as second_hash:
            conversations = composer.create_dataset()

        assert isinstance(composer.loader, HCuaPerfFileLoader)
        assert {c.session_id: len(c.turns) for c in conversations} == SESSION_TURNS
        second_hash.assert_not_called()

    def test_missing_manifest_is_rejected_at_construction(self, tmp_path: Path) -> None:
        trace = build(tmp_path, "h_cua.jsonl.zst")
        (tmp_path / "h_cua.meta.json").unlink()

        with pytest.raises(DatasetLoaderError, match="h_cua.meta.json"):
            HCuaPerfFileLoader(
                filename=trace, prompt_generator=MagicMock(), run=_run(trace)
            )

    def test_ignore_trace_delays_is_rejected_like_the_hub_loader(
        self, tmp_path: Path
    ) -> None:
        trace = build(tmp_path, "h_cua.jsonl.zst")

        with pytest.raises(DatasetLoaderError, match="--ignore-trace-delays"):
            HCuaPerfFileLoader(
                filename=trace,
                prompt_generator=MagicMock(),
                run=_run(trace, ignore_trace_delays=True),
            )

    @pytest.mark.parametrize(
        "sha256, match",
        [
            param({}, "no sha256", id="trace_not_named"),
            param({"h_cua.jsonl.zst": "0" * 64}, "does not match", id="stale_digest"),
        ],
    )  # fmt: skip
    def test_a_manifest_not_naming_the_trace_digest_is_refused(
        self, tmp_path: Path, sha256: dict[str, str], match: str
    ) -> None:
        trace = build(tmp_path, "h_cua.jsonl.zst")
        manifest = tmp_path / "h_cua.meta.json"
        meta = orjson.loads(manifest.read_bytes()) | {"sha256": sha256}
        manifest.write_bytes(orjson.dumps(meta))
        loader = HCuaPerfFileLoader(
            filename=trace, prompt_generator=MagicMock(), run=_run(trace)
        )

        with pytest.raises(DatasetLoaderError, match=match):
            loader.load_dataset()
