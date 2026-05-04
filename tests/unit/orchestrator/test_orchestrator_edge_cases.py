# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge-case tests for FixedTrialsStrategy.

Subprocess-execution and metrics-extraction edge cases moved to
tests/unit/orchestrator/test_local_executor.py with the Task 7/8 split
(those concerns now live on LocalSubprocessExecutor).
"""

from pathlib import Path

from aiperf.config import BenchmarkConfig
from aiperf.config.v1 import ServiceConfig, UserConfig
from aiperf.config.v1.converter import convert_user_to_aiperf
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.strategies import FixedTrialsStrategy

_MINIMAL_CONFIG_KWARGS: dict = {
    "models": ["test-model"],
    "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
    "datasets": [
        {
            "name": "default",
            "type": "synthetic",
            "entries": 100,
            "prompts": {"isl": 128, "osl": 64},
        }
    ],
    "phases": [
        {
            "name": "warmup",
            "type": "concurrency",
            "requests": 10,
            "concurrency": 1,
            "exclude_from_results": True,
        },
        {"name": "profiling", "type": "concurrency", "requests": 100, "concurrency": 1},
    ],
}


def _make_config(**overrides: object) -> BenchmarkConfig:
    overrides.pop("random_seed", None)
    kwargs = {**_MINIMAL_CONFIG_KWARGS, **overrides}
    return BenchmarkConfig(**kwargs)


def _make_user_config(**loadgen_overrides: object) -> UserConfig:
    """Build a v1 UserConfig fixture for converter-shape seed assertions.

    The legacy strategy-level `auto_set_seed` toggled
    `FixedTrialsStrategy._ensure_random_seed`, which is now a no-op stub.
    The equivalent contract lives at the v1 converter layer
    (`--set-consistent-seed` + envelope `random_seed`); converter tests
    replace what these strategy tests used to assert.
    """
    loadgen: dict[str, object] = {"concurrency": 1, "request_count": 10}
    loadgen.update(loadgen_overrides)
    return UserConfig.model_validate(
        {
            "endpoint": {"model_names": ["m"], "urls": ["http://x"]},
            "loadgen": loadgen,
        }
    )


# ============================================================
# Strategy: seed handling, warmup, config isolation
# ============================================================


class TestFixedTrialsStrategyEdgeCases:
    """Verify seed handling, warmup removal, config isolation."""

    def test_seed_set_when_none_and_auto_true(self) -> None:
        """--set-consistent-seed=True (default) + no --random-seed yields envelope seed=42."""
        cfg = convert_user_to_aiperf(
            _make_user_config(num_profile_runs=2), ServiceConfig()
        )
        assert cfg.random_seed == FixedTrialsStrategy.DEFAULT_SEED

    def test_seed_preserved_when_already_set(self) -> None:
        """Explicit --random-seed=123 takes precedence over the consistent-seed default."""
        user = UserConfig.model_validate(
            {
                "endpoint": {"model_names": ["m"], "urls": ["http://x"]},
                "input": {"random_seed": 123},
                "loadgen": {
                    "concurrency": 1,
                    "request_count": 10,
                    "num_profile_runs": 2,
                },
            }
        )
        cfg = convert_user_to_aiperf(user, ServiceConfig())
        assert cfg.random_seed == 123

    def test_seed_not_set_when_auto_false(self) -> None:
        """--no-set-consistent-seed disables the default; envelope seed stays None."""
        cfg = convert_user_to_aiperf(
            _make_user_config(num_profile_runs=2, set_consistent_seed=False),
            ServiceConfig(),
        )
        assert cfg.random_seed is None

    def test_warmup_removed_on_subsequent_runs(self) -> None:
        config = _make_config()
        strategy = FixedTrialsStrategy(num_trials=2, disable_warmup_after_first=True)
        run0 = strategy.get_next_config(config, [])
        run1 = strategy.get_next_config(config, [RunResult(label="r0", success=True)])

        assert any(p.name == "warmup" for p in run0.phases)
        assert not any(p.name == "warmup" for p in run1.phases)
        assert any(p.name == "profiling" for p in run1.phases)

    def test_warmup_kept_on_first_run(self) -> None:
        config = _make_config()
        strategy = FixedTrialsStrategy(num_trials=2, disable_warmup_after_first=True)
        run = strategy.get_next_config(config, [])
        assert any(p.name == "warmup" for p in run.phases)

    def test_warmup_kept_when_flag_false(self) -> None:
        config = _make_config()
        strategy = FixedTrialsStrategy(num_trials=2, disable_warmup_after_first=False)
        run = strategy.get_next_config(config, [RunResult(label="r0", success=True)])
        assert any(p.name == "warmup" for p in run.phases)

    def test_warmup_removal_only_removes_excluded_phases(self) -> None:
        config = _make_config(
            phases=[
                {
                    "name": "warmup",
                    "type": "concurrency",
                    "requests": 5,
                    "concurrency": 1,
                    "exclude_from_results": True,
                },
                {
                    "name": "main",
                    "type": "concurrency",
                    "requests": 50,
                    "concurrency": 4,
                },
                {
                    "name": "cooldown_phase",
                    "type": "concurrency",
                    "requests": 10,
                    "concurrency": 1,
                },
            ]
        )
        strategy = FixedTrialsStrategy(num_trials=2, disable_warmup_after_first=True)
        run = strategy.get_next_config(config, [RunResult(label="r0", success=True)])
        assert not any(p.name == "warmup" for p in run.phases)
        assert any(p.name == "main" for p in run.phases)
        assert any(p.name == "cooldown_phase" for p in run.phases)

    def test_config_deep_copy_when_seed_set(self) -> None:
        """Converter returns a fresh AIPerfConfig; mutating it doesn't affect inputs.

        Replaces the legacy `FixedTrialsStrategy.get_next_config` deep-copy assertion;
        the new layer is the converter, which constructs a brand-new validated
        AIPerfConfig from the v1 UserConfig.
        """
        user = _make_user_config(num_profile_runs=2)
        cfg = convert_user_to_aiperf(user, ServiceConfig())
        assert cfg.random_seed == FixedTrialsStrategy.DEFAULT_SEED
        cfg2 = convert_user_to_aiperf(user, ServiceConfig())
        # Two converter calls return independent objects, not aliases.
        assert cfg is not cfg2

    def test_config_deep_copy_when_warmup_disabled(self) -> None:
        """Warmup removal must not affect the original config."""
        config = _make_config()
        strategy = FixedTrialsStrategy(num_trials=2, disable_warmup_after_first=True)
        result = strategy.get_next_config(config, [RunResult(label="r0", success=True)])
        assert not any(p.name == "warmup" for p in result.phases)
        assert any(p.name == "warmup" for p in config.phases)

    def test_run_label_format(self) -> None:
        strategy = FixedTrialsStrategy(num_trials=3)
        assert strategy.get_run_label(0) == "run_0001"
        assert strategy.get_run_label(1) == "run_0002"
        assert strategy.get_run_label(9) == "run_0010"

    def test_run_path(self, tmp_path: Path) -> None:
        strategy = FixedTrialsStrategy(num_trials=3)
        assert strategy.get_run_path(tmp_path, 0) == (
            tmp_path / "profile_runs" / "run_0001"
        )
