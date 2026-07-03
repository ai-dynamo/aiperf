# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the BurstGPT trace loader's synthesis gate.

The loader shares the ``synthesis_should_apply`` helper with the other trace
loaders. Before the fix, its inline gate omitted ``prompt_len_multiplier`` and
``output_len_multiplier``, so those flags silently no-opped. Real config
objects only (not MagicMock, which hides missing-field drift).
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from pytest import param

from aiperf.config import AIPerfConfig
from aiperf.dataset.loader.burst_gpt import BurstGPTTraceDatasetLoader
from tests.unit.dataset.loader.conftest import _make_run

_BASE = dict(
    models=["test-model"],
    endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
    phases=[
        {"name": "default", "type": "concurrency", "requests": 10, "concurrency": 1}
    ],
)

_CSV_ROWS = [
    "Timestamp,Request tokens,Response tokens",
    "0.0,472,18",
    "0.1,1087,230",
]


def _make_burst_config(path: str, synthesis: dict | None) -> AIPerfConfig:
    dataset: dict = {"type": "file", "path": path, "format": "burst_gpt_trace"}
    if synthesis is not None:
        dataset["synthesis"] = synthesis
    return AIPerfConfig(
        benchmark={**_BASE, "datasets": [{"name": "default", **dataset}]}
    )


@pytest.fixture
def mock_prompt_generator():
    generator = Mock()
    generator.generate.return_value = "Generated prompt text"
    generator._decoded_cache = {}
    generator._build_token_sequence.return_value = [1, 2, 3, 4, 5]
    return generator


def _loader(tmp_path: Path, mock_prompt_generator, synthesis: dict | None):
    csv_file = tmp_path / "burst.csv"
    csv_file.write_text("\n".join(_CSV_ROWS) + "\n")
    config = _make_burst_config(str(csv_file), synthesis)
    return BurstGPTTraceDatasetLoader(
        filename=str(csv_file),
        run=_make_run(config),
        prompt_generator=mock_prompt_generator,
    )


class TestBurstGPTSynthesisGate:
    """``_apply_synthesis`` fires iff a transform multiplier is non-default."""

    @pytest.mark.parametrize(
        "overrides",
        [
            param({"prompt_len_multiplier": 2.0}, id="prompt_len_multiplier"),
            param({"output_len_multiplier": 2.0}, id="output_len_multiplier"),
            param({"speedup_ratio": 2.0}, id="speedup_ratio"),
        ],
    )  # fmt: skip
    def test_lone_transform_triggers_apply(
        self, tmp_path, mock_prompt_generator, overrides
    ):
        loader = _loader(tmp_path, mock_prompt_generator, overrides)
        with patch.object(loader, "_apply_synthesis", side_effect=lambda d: d) as spy:
            loader.load_dataset()
        spy.assert_called_once()

    def test_default_synthesis_does_not_apply(self, tmp_path, mock_prompt_generator):
        loader = _loader(tmp_path, mock_prompt_generator, {})
        with patch.object(loader, "_apply_synthesis", side_effect=lambda d: d) as spy:
            loader.load_dataset()
        spy.assert_not_called()
