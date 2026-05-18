# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the expanded CLI magic-list set and `--sweep-type` flag.

Companion to ``test_cli_magic_list_sugar.py`` (which covered the initial
``--concurrency`` / ``--prefill-concurrency`` / ``--request-rate`` set).
This file covers the second wave: ``--isl``, ``--osl``,
``--request-count``, ``--benchmark-duration``, ``--num-users``,
``--num-conversations``, plus the cross-cutting ``--sweep-type``
{grid,zip} switch.
"""

from __future__ import annotations

import pytest

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf
from aiperf.config.sweep import GridSweep, ZipSweep


class TestDatasetMagicLists:
    """`--isl` / `--osl` hoist directly to dataset-rooted sweep params."""

    def test_isl_comma_list_promotes_to_dataset_sweep(self):
        cli = CLIConfig(model_names=["m"], prompt_input_tokens_mean="128,512,2048")
        cfg = convert_cli_to_aiperf(cli)
        assert isinstance(cfg.sweep, GridSweep)
        assert cfg.sweep.parameters == {
            "datasets.main.prompts.isl.mean": [128, 512, 2048]
        }

    def test_osl_comma_list_promotes_to_dataset_sweep(self):
        cli = CLIConfig(model_names=["m"], prompt_output_tokens_mean="64,128,256")
        cfg = convert_cli_to_aiperf(cli)
        assert cfg.sweep.parameters == {
            "datasets.main.prompts.osl.mean": [64, 128, 256]
        }

    def test_isl_osl_grid_cross_product(self):
        cli = CLIConfig(
            model_names=["m"],
            prompt_input_tokens_mean="128,512",
            prompt_output_tokens_mean="64,256",
        )
        cfg = convert_cli_to_aiperf(cli)
        assert isinstance(cfg.sweep, GridSweep)
        assert cfg.sweep.parameters == {
            "datasets.main.prompts.isl.mean": [128, 512],
            "datasets.main.prompts.osl.mean": [64, 256],
        }

    def test_isl_scalar_does_not_create_sweep(self):
        cli = CLIConfig(model_names=["m"], prompt_input_tokens_mean=512)
        cfg = convert_cli_to_aiperf(cli)
        assert cfg.sweep is None

    def test_isl_placeholder_lands_on_base_dataset(self):
        # First list element becomes the base config's isl.mean so
        # AIPerfConfig validates; each variation overrides per-cell.
        cli = CLIConfig(model_names=["m"], prompt_input_tokens_mean="128,512,2048")
        cfg = convert_cli_to_aiperf(cli)
        main = next(d for d in cfg.benchmark.datasets if d.name == "main")
        assert main.prompts.isl.mean == 128


class TestPhaseMagicLists:
    """request_count / benchmark_duration / conversation_num / num_users."""

    def test_request_count_list_sweeps_phases_profiling_requests(self):
        cli = CLIConfig(model_names=["m"], request_count="100,500,1000")
        cfg = convert_cli_to_aiperf(cli)
        assert cfg.sweep.parameters == {"phases.profiling.requests": [100, 500, 1000]}

    def test_benchmark_duration_list_sweeps_phases_profiling_duration(self):
        cli = CLIConfig(model_names=["m"], benchmark_duration="30,60,120")
        cfg = convert_cli_to_aiperf(cli)
        assert cfg.sweep.parameters == {
            "phases.profiling.duration": [30.0, 60.0, 120.0]
        }

    def test_num_users_list_sweeps_phases_profiling_users(self):
        # num_users requires --user-centric-rate to land on UserCentricPhase.
        cli = CLIConfig(
            model_names=["m"],
            user_centric_rate=10.0,
            conversation_turn_mean=3,
            num_users="4,8,16",
        )
        cfg = convert_cli_to_aiperf(cli)
        assert cfg.sweep.parameters == {"phases.profiling.users": [4, 8, 16]}

    def test_conversation_num_list_sweeps_phase_sessions(self):
        cli = CLIConfig(model_names=["m"], conversation_num="50,100,200")
        cfg = convert_cli_to_aiperf(cli)
        assert cfg.sweep.parameters == {"phases.profiling.sessions": [50, 100, 200]}

    def test_conversation_num_list_sizes_dataset_to_max(self):
        # Phase.sessions varies per variation, but the dataset entries pool
        # is sized to max(list) so every variation has its unique-session set.
        cli = CLIConfig(model_names=["m"], conversation_num="50,100,200")
        cfg = convert_cli_to_aiperf(cli)
        main = next(d for d in cfg.benchmark.datasets if d.name == "main")
        assert main.entries == 200

    def test_combined_grid_across_phase_and_dataset(self):
        cli = CLIConfig(
            model_names=["m"],
            request_count="100,500",
            prompt_input_tokens_mean="128,512",
        )
        cfg = convert_cli_to_aiperf(cli)
        assert isinstance(cfg.sweep, GridSweep)
        assert cfg.sweep.parameters == {
            "phases.profiling.requests": [100, 500],
            "datasets.main.prompts.isl.mean": [128, 512],
        }


class TestSweepTypeZip:
    """`--sweep-type zip` switches the promote pass to emit a zip sweep block."""

    def test_zip_with_equal_length_lists_succeeds(self):
        cli = CLIConfig(
            model_names=["m"],
            sweep_type="zip",
            prompt_input_tokens_mean="128,512,2048",
            prompt_output_tokens_mean="128,256,512",
            concurrency="4,16,64",
        )
        cfg = convert_cli_to_aiperf(cli)
        assert isinstance(cfg.sweep, ZipSweep)
        assert cfg.sweep.parameters == {
            "datasets.main.prompts.isl.mean": [128, 512, 2048],
            "datasets.main.prompts.osl.mean": [128, 256, 512],
            "phases.profiling.concurrency": [4, 16, 64],
        }

    def test_zip_default_is_grid(self):
        cli = CLIConfig(
            model_names=["m"],
            prompt_input_tokens_mean="128,512",
            prompt_output_tokens_mean="64,256",
        )
        cfg = convert_cli_to_aiperf(cli)
        assert isinstance(cfg.sweep, GridSweep)

    def test_zip_with_mismatched_lengths_rejected(self):
        # `_expand_zip_sweep` enforces equal length downstream.
        cli = CLIConfig(
            model_names=["m"],
            sweep_type="zip",
            prompt_input_tokens_mean="128,512,2048",
            concurrency="4,16",  # length 2 vs 3
        )
        with pytest.raises(Exception, match="equal length|same length"):
            cfg = convert_cli_to_aiperf(cli)
            # Length check fires at expand time (not at base config validation).
            from aiperf.config.sweep import expand_sweep

            expand_sweep(cfg.model_dump(by_alias=False, exclude_none=False))

    def test_zip_with_single_magic_list_still_works(self):
        # zip topology with one list is just a 1-dim sweep, length=N.
        cli = CLIConfig(model_names=["m"], sweep_type="zip", concurrency="4,8,16")
        cfg = convert_cli_to_aiperf(cli)
        assert isinstance(cfg.sweep, ZipSweep)
        assert cfg.sweep.parameters == {"phases.profiling.concurrency": [4, 8, 16]}


class TestRealisticTrafficShapesViaZip:
    """The killer demo: paired ISL/OSL + their stddevs in zip mode."""

    def test_isl_stddev_list_promotes(self):
        cli = CLIConfig(model_names=["m"], prompt_input_tokens_stddev="10,50,200")
        cfg = convert_cli_to_aiperf(cli)
        assert cfg.sweep.parameters == {
            "datasets.main.prompts.isl.stddev": [10.0, 50.0, 200.0]
        }

    def test_osl_stddev_list_promotes(self):
        cli = CLIConfig(
            model_names=["m"],
            prompt_output_tokens_mean=256,  # required so OSL block is emitted
            prompt_output_tokens_stddev="5,25,100",
        )
        cfg = convert_cli_to_aiperf(cli)
        assert cfg.sweep.parameters == {
            "datasets.main.prompts.osl.stddev": [5.0, 25.0, 100.0]
        }

    def test_conversation_turn_mean_list_promotes(self):
        cli = CLIConfig(model_names=["m"], conversation_turn_mean="1,3,8")
        cfg = convert_cli_to_aiperf(cli)
        assert cfg.sweep.parameters == {"datasets.main.turns.mean": [1, 3, 8]}

    def test_zip_paired_isl_osl_with_stddevs(self):
        # The reason this whole feature exists: realistic small/medium/large
        # traffic shapes co-vary mean AND stddev. Zip pairs them lockstep.
        cli = CLIConfig(
            model_names=["m"],
            sweep_type="zip",
            prompt_input_tokens_mean="128,512,2048",
            prompt_input_tokens_stddev="10,50,200",
            prompt_output_tokens_mean="64,256,1024",
            prompt_output_tokens_stddev="5,25,100",
        )
        cfg = convert_cli_to_aiperf(cli)
        assert isinstance(cfg.sweep, ZipSweep)
        assert cfg.sweep.parameters == {
            "datasets.main.prompts.isl.mean": [128, 512, 2048],
            "datasets.main.prompts.isl.stddev": [10.0, 50.0, 200.0],
            "datasets.main.prompts.osl.mean": [64, 256, 1024],
            "datasets.main.prompts.osl.stddev": [5.0, 25.0, 100.0],
        }

    def test_isl_stddev_placeholder_lands_on_base(self):
        cli = CLIConfig(model_names=["m"], prompt_input_tokens_stddev="10,50,200")
        cfg = convert_cli_to_aiperf(cli)
        main = next(d for d in cfg.benchmark.datasets if d.name == "main")
        assert main.prompts.isl.stddev == 10.0  # first element

    def test_turn_mean_placeholder_lands_on_base(self):
        cli = CLIConfig(model_names=["m"], conversation_turn_mean="1,3,8")
        cfg = convert_cli_to_aiperf(cli)
        main = next(d for d in cfg.benchmark.datasets if d.name == "main")
        assert main.turns.mean == 1  # first element


class TestConversationTurnMeanUserCentricValidation:
    """Sweep min(list) must satisfy user_centric's turn_mean >= 2 rule."""

    def test_user_centric_with_turn_mean_min_below_2_rejected(self):
        cli = CLIConfig(
            model_names=["m"],
            user_centric_rate=10.0,
            num_users=4,
            conversation_turn_mean="1,3,8",  # min=1 violates >=2
        )
        with pytest.raises(ValueError, match="--session-turns-mean >= 2"):
            convert_cli_to_aiperf(cli)

    def test_user_centric_with_turn_mean_min_at_or_above_2_accepted(self):
        cli = CLIConfig(
            model_names=["m"],
            user_centric_rate=10.0,
            num_users=4,
            conversation_turn_mean="2,3,8",
        )
        cfg = convert_cli_to_aiperf(cli)
        assert cfg.sweep.parameters == {"datasets.main.turns.mean": [2, 3, 8]}

    """Regression: placeholder must land on the base config so AIPerfConfig
    validates even when the magic-list strips required fields.
    """

    def test_request_count_placeholder_on_phase(self):
        cli = CLIConfig(model_names=["m"], request_count="100,500,1000")
        cfg = convert_cli_to_aiperf(cli)
        prof = next(p for p in cfg.benchmark.phases if p.name == "profiling")
        assert prof.requests == 100  # first list element

    def test_benchmark_duration_placeholder_on_phase(self):
        cli = CLIConfig(model_names=["m"], benchmark_duration="30,60,120")
        cfg = convert_cli_to_aiperf(cli)
        prof = next(p for p in cfg.benchmark.phases if p.name == "profiling")
        assert prof.duration == 30.0
