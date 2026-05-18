# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the v1 LoadGeneratorConfig DTO.

The v1 LoadGeneratorConfig is a CLI-only input DTO with no validators
(see `aiperf.config.flags._loadgen` module docstring). The earlier
``confidence_level`` / ``num_profile_runs`` / ``parameter_sweep_*``
validators tested here have moved:

* multi-run knobs (`num_profile_runs`, `confidence_level`,
  `set_consistent_seed`, `profile_run_disable_warmup_after_first`,
  `profile_run_cooldown_seconds`) -> `aiperf.config.sweep.multi_run.MultiRunConfig`
* parameter sweep knobs (`parameter_sweep_mode`,
  `parameter_sweep_cooldown_seconds`, `parameter_sweep_same_seed`,
  `num_profile_runs` for sweeps) -> top-level fields on CLIConfig (sweeping
  section)
  + final validation in AIPerfConfig.

Field-level validation (``ge=1``) on ``concurrency`` list elements is gone
on v1 because the field type is `Any` (BeforeValidator parses raw strings).
The list-shape / duplicates / non-positive checks now live in the v1->v2
converter / AIPerfConfig.

The remaining tests in this file are smoke tests for fields that still live
on `LoadGeneratorConfig` directly with intrinsic Pydantic constraints.
"""

import pytest
from pydantic import ValidationError

from aiperf.config.flags import CLIConfig


class TestConcurrencyDTOFields:
    """Concurrency input shapes accepted by the v1 DTO."""

    def test_single_concurrency_value_succeeds(self) -> None:
        config = CLIConfig(concurrency=10)
        assert config.concurrency == 10

    def test_concurrency_list_valid_values_succeeds(self) -> None:
        config = CLIConfig(concurrency=[10, 20, 30, 40])
        assert config.concurrency == [10, 20, 30, 40]

    def test_concurrency_none_succeeds(self) -> None:
        config = CLIConfig(concurrency=None)
        assert config.concurrency is None

    def test_concurrency_default_is_none(self) -> None:
        config = CLIConfig()
        assert config.concurrency is None


class TestPrefillConcurrencyValidation:
    """`prefill_concurrency` is a CLI magic-list field (`Any` + parser).

    Field-level ``ge=1`` is enforced downstream by ``BasePhaseConfig`` at
    ``convert_cli_to_aiperf`` time, mirroring how ``concurrency`` is
    handled. The DTO itself accepts any int (the parser normalizes
    comma-lists); rejection happens when the value lands on the phase.
    """

    def test_prefill_concurrency_zero_rejected_downstream(self) -> None:
        from aiperf.config.flags.converter import convert_cli_to_aiperf

        with pytest.raises(ValidationError):
            convert_cli_to_aiperf(
                CLIConfig(model_names=["m"], streaming=True, prefill_concurrency=0)
            )

    def test_prefill_concurrency_negative_rejected_downstream(self) -> None:
        from aiperf.config.flags.converter import convert_cli_to_aiperf

        with pytest.raises(ValidationError):
            convert_cli_to_aiperf(
                CLIConfig(model_names=["m"], streaming=True, prefill_concurrency=-1)
            )

    def test_prefill_concurrency_positive_succeeds(self) -> None:
        config = CLIConfig(prefill_concurrency=4)
        assert config.prefill_concurrency == 4
