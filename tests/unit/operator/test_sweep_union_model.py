# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model projection for live and archived sweep records."""

from pathlib import Path
from typing import Any

import orjson
import pytest
from aiperf.operator.sweep_union import (
    _model_from_spec,
    _record_from_archive,
    _record_from_live,
)
from pytest import param


@pytest.mark.parametrize(
    "models, expected",
    [
        param("llama", "llama", id="string"),
        param(["llama", "qwen"], "llama", id="string-list"),
        param([{"name": "llama"}], "llama", id="mapping-list"),
        param({"items": [{"name": "llama"}]}, "llama", id="advanced-items"),
        param({"modelNames": ["llama"]}, "llama", id="advanced-names"),
        param([], None, id="empty"),
    ],
)  # fmt: skip
def test_model_from_spec_accepts_raw_cr_shapes(
    models: Any, expected: str | None
) -> None:
    assert _model_from_spec({"benchmark": {"models": models}}) == expected


def test_record_from_live_reads_benchmark_model() -> None:
    record = _record_from_live(
        {
            "metadata": {"namespace": "ns", "name": "sweep"},
            "spec": {"benchmark": {"models": [{"name": "llama"}]}},
            "status": {"phase": "Running"},
        }
    )

    assert record.model == "llama"


def test_record_from_archive_reads_spec_snapshot_model(tmp_path: Path) -> None:
    epoch_dir = tmp_path / "1714064523"
    epoch_dir.mkdir()
    (epoch_dir / "aggregate.json").write_bytes(
        orjson.dumps(
            {
                "phase": "Succeeded",
                "specSnapshot": {"benchmark": {"models": ["llama"]}},
            }
        )
    )

    record = _record_from_archive("ns", "sweep", epoch_dir)

    assert record is not None
    assert record.model == "llama"
