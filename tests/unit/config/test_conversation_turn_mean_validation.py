# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for `--conversation-turn-mean` >= 1 enforcement.

Before the magic-list refactor the CLI flag was typed `int` with `ge=1`.
After the refactor it became `Any` with a magic-list parser, and the only
downstream bound (`NormalDistribution.mean`) is `ge=0.0` (deliberately,
because OSL=0 and turn_delay mean=0 are legitimate). That left turns
under-validated: `--conversation-turn-mean 0` was silently accepted and
the synthetic composer floors at 1 turn regardless, so users got a
different shape than they asked for. These tests pin the rule that a
turn-count distribution with expected value < 1 is a config error.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.config.dataset import SyntheticDataset
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf


class TestCLIConversationTurnMeanGE1:
    """`--conversation-turn-mean` scalar / list elements must be >= 1."""

    def test_scalar_zero_rejected(self):
        with pytest.raises(
            (ValueError, ValidationError), match=r"turn.*(>=|at least).*1"
        ):
            cli = CLIConfig(model_names=["m"], conversation_turn_mean=0)
            convert_cli_to_aiperf(cli)

    def test_list_with_zero_element_rejected(self):
        with pytest.raises(
            (ValueError, ValidationError), match=r"turn.*(>=|at least).*1"
        ):
            cli = CLIConfig(model_names=["m"], conversation_turn_mean="1,0,4")
            convert_cli_to_aiperf(cli)

    def test_scalar_one_accepted(self):
        cli = CLIConfig(model_names=["m"], conversation_turn_mean=1)
        cfg = convert_cli_to_aiperf(cli)
        main = next(d for d in cfg.benchmark.datasets if d.name == "main")
        assert main.turns is not None
        assert main.turns.expected_value == 1.0

    def test_list_all_ge_1_accepted(self):
        cli = CLIConfig(model_names=["m"], conversation_turn_mean="1,3,8")
        cfg = convert_cli_to_aiperf(cli)
        assert cfg.sweep is not None
        assert cfg.sweep.parameters == {"datasets.main.turns.mean": [1, 3, 8]}


class TestSyntheticDatasetTurnsGE1:
    """Direct Pydantic path (YAML config) must enforce the same rule."""

    @staticmethod
    def _base(turns):
        return {"name": "main", "type": "synthetic", "turns": turns}

    @pytest.mark.parametrize(
        "turns_value",
        [
            param(0, id="scalar-zero"),
            param({"mean": 0}, id="dict-mean-zero"),
            param({"mean": 0.5, "stddev": 0.1}, id="mean-below-one"),
        ],
    )
    def test_turns_expected_value_below_one_rejected(self, turns_value):
        with pytest.raises(ValidationError, match=r"turn.*(>=|at least).*1"):
            SyntheticDataset.model_validate(self._base(turns_value))

    @pytest.mark.parametrize(
        "turns_value",
        [
            param(1, id="scalar-one"),
            param(3, id="scalar-three"),
            param({"mean": 1, "stddev": 0.5}, id="mean-one-with-stddev"),
            param({"mean": 5, "stddev": 2}, id="mean-five"),
        ],
    )
    def test_turns_expected_value_at_or_above_one_accepted(self, turns_value):
        ds = SyntheticDataset.model_validate(self._base(turns_value))
        assert ds.turns is not None
        assert ds.turns.expected_value >= 1.0

    def test_turns_unset_is_allowed(self):
        ds = SyntheticDataset.model_validate({"name": "main", "type": "synthetic"})
        assert ds.turns is None
