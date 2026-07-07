# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy ``sweep.parameters`` spelling: accepted with a deprecation warning.

The envelope restructure renamed the grid/zip variable map from
``parameters:`` to ``variables:``. Upstream YAML using the old spelling must
keep loading (model validation AND raw-dict expansion) rather than failing
with a generic "Field required" error or silently collapsing to one variation.
"""

from __future__ import annotations

import warnings

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.config.sweep import GridSweep, ZipSweep, expand_sweep


def _base_envelope() -> dict:
    return {
        "benchmark": {
            "model": "test-model",
            "endpoint": {"url": "http://localhost:8000", "type": "chat"},
            "phases": [{"name": "profiling", "concurrency": 1, "duration": 10}],
        }
    }


class TestLegacyParametersKeyOnModels:
    @pytest.mark.parametrize(
        "cls, payload",
        [
            param(
                GridSweep,
                {"type": "grid", "parameters": {"concurrency": [1, 2, 4]}},
                id="grid",
            ),
            param(
                ZipSweep,
                {"type": "zip", "parameters": {"concurrency": [1, 2, 4]}},
                id="zip",
            ),
        ],
    )  # fmt: skip
    def test_legacy_parameters_key_loads_with_deprecation_warning(
        self, cls, payload
    ) -> None:
        with pytest.warns(UserWarning, match="deprecated"):
            sweep = cls.model_validate(payload)
        assert sweep.variables == {"concurrency": [1, 2, 4]}

    @pytest.mark.parametrize(
        "cls, sweep_type",
        [
            param(GridSweep, "grid", id="grid"),
            param(ZipSweep, "zip", id="zip"),
        ],
    )  # fmt: skip
    def test_new_variables_key_loads_without_warning(self, cls, sweep_type) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sweep = cls.model_validate(
                {"type": sweep_type, "variables": {"concurrency": [1, 2, 4]}}
            )
        assert sweep.variables == {"concurrency": [1, 2, 4]}

    def test_both_spellings_set_raises_targeted_error(self) -> None:
        with pytest.raises(ValidationError, match="deprecated alias"):
            GridSweep.model_validate(
                {
                    "type": "grid",
                    "variables": {"concurrency": [1, 2]},
                    "parameters": {"duration": [10, 20]},
                }
            )

    def test_legacy_key_does_not_mutate_caller_dict(self) -> None:
        payload = {"type": "grid", "parameters": {"concurrency": [1, 2]}}
        with pytest.warns(UserWarning, match="deprecated"):
            GridSweep.model_validate(payload)
        assert "parameters" in payload
        assert "variables" not in payload


class TestLegacyParametersKeyOnExpansion:
    @pytest.mark.parametrize(
        "sweep_type, expected_count",
        [
            param("grid", 3, id="grid"),
            param("zip", 3, id="zip"),
        ],
    )  # fmt: skip
    def test_expand_sweep_legacy_parameters_produces_all_variations(
        self, sweep_type: str, expected_count: int
    ) -> None:
        data = _base_envelope()
        data["sweep"] = {
            "type": sweep_type,
            "parameters": {"phases.profiling.concurrency": [1, 2, 4]},
        }
        expanded = expand_sweep(data)
        assert len(expanded) == expected_count
        swept = [v.values["phases.profiling.concurrency"] for _, v in expanded]
        assert swept == [1, 2, 4]

    def test_expand_sweep_new_variables_key_wins_when_present(self) -> None:
        data = _base_envelope()
        data["sweep"] = {
            "type": "grid",
            "variables": {"phases.profiling.concurrency": [1, 2]},
        }
        expanded = expand_sweep(data)
        assert len(expanded) == 2
