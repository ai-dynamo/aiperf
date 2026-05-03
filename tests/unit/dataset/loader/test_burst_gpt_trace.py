# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.dataset.loader.burst_gpt import BurstGPTTraceDatasetLoader
from aiperf.dataset.loader.models import BurstGPTTrace

# ============================================================================
# Helpers
# ============================================================================

_CSV_HEADER = [
    "Timestamp",
    "Model",
    "Request tokens",
    "Response tokens",
    "Total tokens",
    "Log Type",
]


def _make_csv_file(
    rows: list[dict], tmp_path: Path, filename: str = "burst_gpt.csv"
) -> str:
    path = tmp_path / filename
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def _make_loader(
    filename: str, user_config: UserConfig | None = None
) -> BurstGPTTraceDatasetLoader:
    user_config = user_config or UserConfig(
        endpoint=EndpointConfig(model_names=["test-model"])
    )
    prompt_generator = Mock()
    prompt_generator.generate.return_value = "Generated prompt"
    prompt_generator._decoded_cache = {}
    prompt_generator._build_token_sequence.return_value = [1, 2, 3]
    return BurstGPTTraceDatasetLoader(
        filename=filename,
        user_config=user_config,
        prompt_generator=prompt_generator,
    )


def _make_row(
    timestamp: str = "0.1",
    request_tokens: str = "100",
    response_tokens: str = "40",
    model: str = "gpt-4",
) -> dict:
    return {
        "Timestamp": timestamp,
        "Model": model,
        "Request tokens": request_tokens,
        "Response tokens": response_tokens,
        "Total tokens": str(int(request_tokens) + int(response_tokens)),
        "Log Type": "chat",
    }


# ============================================================================
# BurstGPTTrace Model Tests
# ============================================================================


class TestBurstGPTTrace:
    def test_create_valid(self) -> None:
        trace = BurstGPTTrace(
            timestamp=1.5,
            input_length=512,
            output_length=128,
        )
        assert trace.timestamp == 1.5
        assert trace.input_length == 512
        assert trace.output_length == 128

    def test_missing_timestamp_raises(self) -> None:
        with pytest.raises(ValidationError, match="timestamp"):
            BurstGPTTrace(input_length=10, output_length=5)

    def test_missing_input_length_raises(self) -> None:
        with pytest.raises(ValidationError, match="input_length"):
            BurstGPTTrace(timestamp=1.0, output_length=5)

    def test_missing_output_length_raises(self) -> None:
        with pytest.raises(ValidationError, match="output_length"):
            BurstGPTTrace(timestamp=1.0, input_length=10)


# ============================================================================
# can_load Tests
# ============================================================================


class TestCanLoad:
    def test_valid_csv_returns_true(self, tmp_path: Path) -> None:
        path = _make_csv_file([], tmp_path)
        assert BurstGPTTraceDatasetLoader.can_load(filename=path) is True

    def test_missing_column_returns_false(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.csv"
        with open(path, "w") as f:
            f.write("Timestamp,Request tokens\n")  # missing Response tokens
        assert BurstGPTTraceDatasetLoader.can_load(filename=path) is False

    def test_non_csv_file_returns_false(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        path.write_text('{"input_length": 10}\n')
        assert BurstGPTTraceDatasetLoader.can_load(filename=path) is False

    def test_none_filename_returns_false(self) -> None:
        assert BurstGPTTraceDatasetLoader.can_load(filename=None) is False

    def test_nonexistent_file_returns_false(self, tmp_path: Path) -> None:
        non_existent = tmp_path / "does_not_exist.csv"
        assert BurstGPTTraceDatasetLoader.can_load(filename=str(non_existent)) is False


# ============================================================================
# load_dataset Tests
# ============================================================================


class TestLoadDataset:
    def test_each_row_becomes_its_own_session(self, tmp_path: Path) -> None:
        rows = [_make_row("0.1", "100", "40"), _make_row("0.2", "200", "80")]
        path = _make_csv_file(rows, tmp_path)
        loader = _make_loader(path)
        data = loader.load_dataset()

        # Each row gets a unique session ID
        assert len(data) == 2
        for traces in data.values():
            assert len(traces) == 1

    def test_timestamps_converted_to_milliseconds(self, tmp_path: Path) -> None:
        rows = [_make_row("1.5", "10", "5")]
        path = _make_csv_file(rows, tmp_path)
        loader = _make_loader(path)
        data = loader.load_dataset()

        trace = next(iter(data.values()))[0]
        assert trace.timestamp == 1500.0

    def test_loads_token_counts(self, tmp_path: Path) -> None:
        rows = [_make_row("0.1", "300", "75")]
        path = _make_csv_file(rows, tmp_path)
        loader = _make_loader(path)
        data = loader.load_dataset()

        trace = next(iter(data.values()))[0]
        assert trace.input_length == 300
        assert trace.output_length == 75

    def test_skips_rows_with_bad_timestamp(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(_CSV_HEADER)
            writer.writerow(["not_a_float", "gpt-4", "100", "40", "140", "chat"])
            writer.writerow(["0.3", "gpt-4", "100", "40", "140", "chat"])  # valid
        loader = _make_loader(str(path))
        data = loader.load_dataset()

        assert len(data) == 1

    def test_skips_rows_with_bad_token_count(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(_CSV_HEADER)
            writer.writerow(["0.2", "gpt-4", "not_an_int", "40", "40", "chat"])
            writer.writerow(["0.3", "gpt-4", "100", "40", "140", "chat"])  # valid
        loader = _make_loader(str(path))
        data = loader.load_dataset()

        assert len(data) == 1


# ============================================================================
# Synthesis Hook Tests
# ============================================================================


class TestSynthesisHooks:
    def test_synthesis_exclude_fields_is_empty(self, tmp_path: Path) -> None:
        path = _make_csv_file([], tmp_path)
        loader = _make_loader(path)
        assert loader._synthesis_exclude_fields() == frozenset()

    def test_reconstruct_traces(self, tmp_path: Path) -> None:
        path = _make_csv_file([], tmp_path)
        loader = _make_loader(path)
        originals = [
            BurstGPTTrace(timestamp=100.0, input_length=100, output_length=40),
        ]
        synth_dicts = [{"timestamp": 200.0, "input_length": 150, "output_length": 60}]
        result = loader._reconstruct_traces(originals, synth_dicts)

        assert len(result) == 1
        assert result[0].timestamp == 200.0
        assert result[0].input_length == 150
        assert result[0].output_length == 60

    def test_reconstruct_raises_on_length_mismatch(self, tmp_path: Path) -> None:
        path = _make_csv_file([], tmp_path)
        loader = _make_loader(path)
        originals = [
            BurstGPTTrace(timestamp=100.0, input_length=100, output_length=40),
        ]
        synth_dicts = [
            {"timestamp": 200.0, "input_length": 150, "output_length": 60},
            {"timestamp": 300.0, "input_length": 180, "output_length": 70},
        ]
        with pytest.raises(ValueError, match="synth_dicts length"):
            loader._reconstruct_traces(originals, synth_dicts)
