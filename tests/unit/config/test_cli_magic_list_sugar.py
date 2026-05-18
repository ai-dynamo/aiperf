# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for `--prefill-concurrency` and `--request-rate` CLI magic-list sugar.

Mirrors the long-standing `--concurrency 1,2,4` behavior. Each flag now
accepts a comma-separated list at the CLI which is promoted by
`_promote_magic_lists_to_sweep_block` to a grid sweep on
`phases.profiling.<field>` before AIPerfConfig validation.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf
from aiperf.config.loader.parsing import (
    parse_float_or_float_list,
    parse_int_or_int_list,
)
from aiperf.config.sweep import GridSweep


class TestParseFloatOrFloatList:
    """Direct unit tests on `parse_float_or_float_list`."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            param(None, None, id="none-passthrough"),
            param(15.5, 15.5, id="float-scalar"),
            param(10, 10.0, id="int-scalar-promoted-to-float"),
            param("10", 10.0, id="numeric-string-scalar"),
            param("10.5", 10.5, id="float-string-scalar"),
            param("10,20,30", [10.0, 20.0, 30.0], id="comma-list-string"),
            param("10.5,20.25", [10.5, 20.25], id="float-comma-list"),
            param("10,", 10.0, id="trailing-comma-collapses-to-scalar"),
            param([1, 2, 3], [1.0, 2.0, 3.0], id="explicit-list"),
            param([5], 5.0, id="single-elem-list-collapses"),
        ],
    )
    def test_valid_inputs(self, raw, expected):
        assert parse_float_or_float_list(raw) == expected

    def test_bool_rejected(self):
        with pytest.raises(TypeError, match="got bool"):
            parse_float_or_float_list(True)

    def test_unparseable_type_raises(self):
        with pytest.raises(TypeError):
            parse_float_or_float_list({"not": "valid"})


class TestPrefillConcurrencyMagicList:
    """`--prefill-concurrency 1,2,4` -> grid sweep on `phases.profiling.prefill_concurrency`."""

    def test_comma_list_promotes_to_grid_sweep(self):
        cli = CLIConfig(
            model_names=["m"],
            streaming=True,
            concurrency=8,
            prefill_concurrency="1,2,4",
        )
        cfg = convert_cli_to_aiperf(cli)
        assert isinstance(cfg.sweep, GridSweep)
        assert cfg.sweep.parameters == {
            "phases.profiling.prefill_concurrency": [1, 2, 4]
        }

    def test_scalar_value_does_not_create_sweep(self):
        cli = CLIConfig(
            model_names=["m"],
            streaming=True,
            concurrency=8,
            prefill_concurrency=4,
        )
        cfg = convert_cli_to_aiperf(cli)
        assert cfg.sweep is None
        # Scalar landed on the phase as-is.
        prof = next(p for p in cfg.benchmark.phases if p.name == "profiling")
        assert prof.prefill_concurrency == 4

    def test_combined_with_concurrency_magic_list(self):
        # Both flags as lists -> cross-product sweep (2 prefill x 2 conc = 4).
        cli = CLIConfig(
            model_names=["m"],
            streaming=True,
            concurrency="4,8",
            prefill_concurrency="1,2",
        )
        cfg = convert_cli_to_aiperf(cli)
        assert isinstance(cfg.sweep, GridSweep)
        assert cfg.sweep.parameters == {
            "phases.profiling.concurrency": [4, 8],
            "phases.profiling.prefill_concurrency": [1, 2],
        }


class TestRequestRateMagicList:
    """`--request-rate 10,20,30` -> grid sweep on `phases.profiling.rate`.

    Naming note: the CLI attribute is `request_rate` but the phase field
    is `rate` (mapped by `_LOADGEN_PHASE_FIELD_MAP`). The promoted sweep
    path uses the phase field name, matching grid-sweep convention.
    """

    def test_comma_list_promotes_to_grid_sweep(self):
        cli = CLIConfig(model_names=["m"], request_rate="10,20,30")
        cfg = convert_cli_to_aiperf(cli)
        assert isinstance(cfg.sweep, GridSweep)
        assert cfg.sweep.parameters == {"phases.profiling.rate": [10.0, 20.0, 30.0]}

    def test_fractional_rates_in_list(self):
        cli = CLIConfig(model_names=["m"], request_rate="0.5,1.5,2.25")
        cfg = convert_cli_to_aiperf(cli)
        assert cfg.sweep.parameters == {"phases.profiling.rate": [0.5, 1.5, 2.25]}

    def test_scalar_value_does_not_create_sweep(self):
        cli = CLIConfig(model_names=["m"], request_rate=15.5)
        cfg = convert_cli_to_aiperf(cli)
        assert cfg.sweep is None
        prof = next(p for p in cfg.benchmark.phases if p.name == "profiling")
        assert prof.rate == 15.5

    def test_required_phase_field_placeholder_preserves_base_validity(self):
        # Regression: PoissonPhase requires `rate` (no default). When the
        # magic-list promote strips the rate list off the phase, validation
        # would fail. The promote now leaves the first element behind as a
        # placeholder so base-config validation passes; each variation
        # overrides per-cell at expand time.
        cli = CLIConfig(model_names=["m"], request_rate="10,20,30")
        cfg = convert_cli_to_aiperf(cli)  # would raise without the fix
        prof = next(p for p in cfg.benchmark.phases if p.name == "profiling")
        assert prof.rate == 10.0  # placeholder = first list element

    def test_combined_with_concurrency_magic_list(self):
        cli = CLIConfig(
            model_names=["m"],
            concurrency="4,8",
            request_rate="10,20",
        )
        cfg = convert_cli_to_aiperf(cli)
        assert isinstance(cfg.sweep, GridSweep)
        assert cfg.sweep.parameters == {
            "phases.profiling.concurrency": [4, 8],
            "phases.profiling.rate": [10.0, 20.0],
        }


class TestParseIntOrIntList:
    """Sanity for the existing int parser still works for prefill_concurrency."""

    def test_comma_list(self):
        assert parse_int_or_int_list("1,2,4") == [1, 2, 4]

    def test_scalar(self):
        assert parse_int_or_int_list("8") == 8

    def test_single_element_collapses(self):
        assert parse_int_or_int_list("8,") == 8
