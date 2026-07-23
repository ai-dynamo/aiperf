# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``--max-context-length`` is Weka-only (recorded peak filter-then-cap).

Docs previously claimed DatasetManager tokenizes and drops oversized
conversations for all formats. Only Weka loaders consume the field; reject
loudly on unsupported datasets so the flag cannot silently no-op.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.common.enums import DatasetFormat
from aiperf.config.dataset.config import FileDataset, PublicDataset
from aiperf.config.flags._converter_dataset import build_dataset
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf
from aiperf.plugin.enums import CustomDatasetType, PublicDatasetType

_WEKA_HF = PublicDatasetType.SEMIANALYSIS_CC_TRACES_WEKA_WITH_SUBAGENTS


@pytest.fixture
def trace_jsonl(tmp_path: Path) -> Path:
    p = tmp_path / "trace.jsonl"
    p.touch()
    return p


class TestMaxContextLengthRouting:
    def test_routes_onto_weka_file(self, trace_jsonl: Path) -> None:
        cli = CLIConfig(
            model_names=["m"],
            input_file=str(trace_jsonl),
            custom_dataset_type=CustomDatasetType.WEKA_TRACE,
            max_context_length=128000,
        )
        out = build_dataset(cli)
        assert out["max_context_length"] == 128000
        ds = convert_cli_to_aiperf(cli).benchmark.datasets[0]
        assert ds.max_context_length == 128000

    def test_routes_onto_weka_public(self) -> None:
        cli = CLIConfig(
            model_names=["m"],
            public_dataset=_WEKA_HF,
            max_context_length=128000,
        )
        out = build_dataset(cli)
        assert out["max_context_length"] == 128000
        ds = convert_cli_to_aiperf(cli).benchmark.datasets[0]
        assert ds.max_context_length == 128000


class TestMaxContextLengthConverterGuard:
    @pytest.mark.parametrize(
        "fmt",
        [
            param(CustomDatasetType.MOONCAKE_TRACE, id="mooncake"),
            param(CustomDatasetType.BASETEN_TRACE, id="baseten"),
            param(CustomDatasetType.SINGLE_TURN, id="single_turn"),
        ],
    )
    def test_rejects_non_weka_file_format(
        self, trace_jsonl: Path, fmt: CustomDatasetType
    ) -> None:
        cli = CLIConfig(
            model_names=["m"],
            input_file=str(trace_jsonl),
            custom_dataset_type=fmt,
            max_context_length=128000,
        )
        with pytest.raises(ValueError, match="only applies to Weka"):
            build_dataset(cli)

    def test_rejects_non_weka_public_dataset(self) -> None:
        cli = CLIConfig(
            model_names=["m"],
            public_dataset=PublicDatasetType.SHAREGPT,
            max_context_length=128000,
        )
        with pytest.raises(ValueError, match="only applies to Weka"):
            build_dataset(cli)

    def test_rejects_synthetic_dataset(self) -> None:
        cli = CLIConfig(
            model_names=["m"],
            prompt_input_tokens_mean=128,
            max_context_length=128000,
        )
        with pytest.raises(ValueError, match="only applies to Weka"):
            build_dataset(cli)


class TestMaxContextLengthModelValidators:
    def test_file_weka_accepts(self, trace_jsonl: Path) -> None:
        ds = FileDataset(
            type="file",
            name="m",
            path=trace_jsonl,
            format=DatasetFormat.WEKA_TRACE,
            max_context_length=128000,
        )
        assert ds.max_context_length == 128000

    def test_file_non_weka_rejects(self, trace_jsonl: Path) -> None:
        with pytest.raises(ValidationError, match="only applies to"):
            FileDataset(
                type="file",
                name="m",
                path=trace_jsonl,
                format=DatasetFormat.MOONCAKE_TRACE,
                max_context_length=128000,
            )

    def test_public_weka_accepts(self) -> None:
        ds = PublicDataset(
            type="public",
            name="m",
            dataset=_WEKA_HF,
            max_context_length=128000,
        )
        assert ds.max_context_length == 128000

    def test_public_non_weka_rejects(self) -> None:
        with pytest.raises(ValidationError, match="only applies to"):
            PublicDataset(
                type="public",
                name="m",
                dataset=PublicDatasetType.SHAREGPT,
                max_context_length=128000,
            )
