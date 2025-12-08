# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for experiment classification functionality in DataLoader.
"""

from pathlib import Path

import pytest

from aiperf.plot.core.data_loader import DataLoader
from aiperf.plot.core.plot_specs import ExperimentClassificationConfig


class TestDataLoaderExperimentClassification:
    """Tests for experiment classification logic."""

    def test_classify_with_baseline_pattern(self, tmp_path: Path) -> None:
        """Test classification with baseline pattern match."""
        config = ExperimentClassificationConfig(
            baselines=["*baseline*"],
            treatments=["*treatment*"],
            default="treatment",
        )
        loader = DataLoader(classification_config=config)

        run_path = tmp_path / "my_baseline_run"
        run_name = "my_baseline_run"

        result = loader._classify_experiment_type(run_path, run_name)
        assert result == "baseline"

    def test_classify_with_treatment_pattern(self, tmp_path: Path) -> None:
        """Test classification with treatment pattern match."""
        config = ExperimentClassificationConfig(
            baselines=["*baseline*"],
            treatments=["*treatment*"],
            default="treatment",
        )
        loader = DataLoader(classification_config=config)

        run_path = tmp_path / "treatment_v1"
        run_name = "treatment_v1"

        result = loader._classify_experiment_type(run_path, run_name)
        assert result == "treatment"

    def test_classify_with_multiple_patterns(self, tmp_path: Path) -> None:
        """Test classification with multiple patterns."""
        config = ExperimentClassificationConfig(
            baselines=["*_agg_*", "*baseline*"],
            treatments=["*_disagg_*", "*treatment*"],
            default="treatment",
        )
        loader = DataLoader(classification_config=config)

        test_cases = [
            ("model_agg_config", "baseline"),
            ("my_baseline", "baseline"),
            ("model_disagg_v2", "treatment"),
            ("treatment_large_io", "treatment"),
        ]

        for run_name, expected in test_cases:
            run_path = tmp_path / run_name
            result = loader._classify_experiment_type(run_path, run_name)
            assert result == expected, f"Failed for {run_name}"

    def test_classify_first_matching_pattern_wins(self, tmp_path: Path) -> None:
        """Test that first matching pattern wins."""
        # baseline patterns checked first
        config = ExperimentClassificationConfig(
            baselines=["*model*"],
            treatments=["*model*"],  # Same pattern
            default="treatment",
        )
        loader = DataLoader(classification_config=config)

        run_path = tmp_path / "model_test"
        run_name = "model_test"

        result = loader._classify_experiment_type(run_path, run_name)
        # baseline patterns are checked first, so should return baseline
        assert result == "baseline"

    def test_classify_uses_default_when_no_match(self, tmp_path: Path) -> None:
        """Test that default is used when no patterns match."""
        config = ExperimentClassificationConfig(
            baselines=["*baseline*"],
            treatments=["*treatment*"],
            default="treatment",
        )
        loader = DataLoader(classification_config=config)

        run_path = tmp_path / "random_run_name"
        run_name = "random_run_name"

        result = loader._classify_experiment_type(run_path, run_name)
        assert result == "treatment"

    def test_classify_uses_baseline_default(self, tmp_path: Path) -> None:
        """Test that default can be set to baseline."""
        config = ExperimentClassificationConfig(
            baselines=["*baseline*"],
            treatments=["*treatment*"],
            default="baseline",
        )
        loader = DataLoader(classification_config=config)

        run_path = tmp_path / "random_run_name"
        run_name = "random_run_name"

        result = loader._classify_experiment_type(run_path, run_name)
        assert result == "baseline"

    def test_classify_without_config(self, tmp_path: Path) -> None:
        """Test classification without config falls back to treatment."""
        loader = DataLoader(classification_config=None)

        run_path = tmp_path / "any_run"
        run_name = "any_run"

        result = loader._classify_experiment_type(run_path, run_name)
        assert result == "treatment"

    def test_classify_matches_full_path(self, tmp_path: Path) -> None:
        """Test that patterns can match against full path."""
        config = ExperimentClassificationConfig(
            baselines=["*/experiment/*baseline*"],
            treatments=["*treatment*"],
            default="treatment",
        )
        loader = DataLoader(classification_config=config)

        run_path = tmp_path / "experiment" / "baseline_run"
        run_name = "baseline_run"

        result = loader._classify_experiment_type(run_path, run_name)
        assert result == "baseline"

    def test_classify_case_sensitive(self, tmp_path: Path) -> None:
        """Test that pattern matching is case-sensitive."""
        config = ExperimentClassificationConfig(
            baselines=["*baseline*"],
            treatments=["*treatment*"],
            default="treatment",
        )
        loader = DataLoader(classification_config=config)

        # Uppercase should not match lowercase pattern
        run_path = tmp_path / "BASELINE_run"
        run_name = "BASELINE_run"

        result = loader._classify_experiment_type(run_path, run_name)
        assert result == "treatment"  # Falls back to default

    def test_extract_experiment_group_parent_matches_baseline(
        self, tmp_path: Path
    ) -> None:
        """Test that runs with parent matching baseline pattern use parent name."""
        config = ExperimentClassificationConfig(
            baselines=["*baseline*"],
            treatments=["*treatment*"],
            default="treatment",
        )
        loader = DataLoader(classification_config=config)

        # Create: baseline/concurrency_1/
        parent = tmp_path / "baseline"
        nested = parent / "concurrency_1"
        nested.mkdir(parents=True)

        result = loader._extract_experiment_group(nested, "concurrency_1")
        assert result == "baseline"

    def test_extract_experiment_group_parent_matches_treatment(
        self, tmp_path: Path
    ) -> None:
        """Test that runs with parent matching treatment pattern use parent name."""
        config = ExperimentClassificationConfig(
            baselines=["*baseline*"],
            treatments=["*treatment*"],
            default="treatment",
        )
        loader = DataLoader(classification_config=config)

        # Create: treatment_1/concurrency_1/
        parent = tmp_path / "treatment_1"
        nested = parent / "concurrency_1"
        nested.mkdir(parents=True)

        result = loader._extract_experiment_group(nested, "concurrency_1")
        assert result == "treatment_1"

    def test_extract_experiment_group_multiple_treatments_separate_groups(
        self, tmp_path: Path
    ) -> None:
        """Test that treatment_1 and treatment_2 are separate groups."""
        config = ExperimentClassificationConfig(
            baselines=["*baseline*"],
            treatments=["*treatment*"],
            default="treatment",
        )
        loader = DataLoader(classification_config=config)

        # Create: treatment_1/run/ and treatment_2/run/
        for i in [1, 2]:
            parent = tmp_path / f"treatment_{i}"
            nested = parent / "run"
            nested.mkdir(parents=True)

            result = loader._extract_experiment_group(nested, "run")
            assert result == f"treatment_{i}"

    def test_extract_experiment_group_parent_no_match_uses_run_name(
        self, tmp_path: Path
    ) -> None:
        """Test that if parent doesn't match any pattern, uses run name."""
        config = ExperimentClassificationConfig(
            baselines=["*baseline*"],
            treatments=["*treatment*"],
            default="treatment",
        )
        loader = DataLoader(classification_config=config)

        # Create: artifacts/random_run/
        parent = tmp_path / "artifacts"
        nested = parent / "random_run"
        nested.mkdir(parents=True)

        result = loader._extract_experiment_group(nested, "random_run")
        assert result == "random_run"  # Parent doesn't match, use run name

    def test_extract_experiment_group_without_config(self, tmp_path: Path) -> None:
        """Test that without classification config, uses run name."""
        loader = DataLoader(classification_config=None)

        # Create: baseline/concurrency_8/
        parent = tmp_path / "baseline"
        nested = parent / "concurrency_8"
        nested.mkdir(parents=True)

        result = loader._extract_experiment_group(nested, "concurrency_8")
        assert result == "concurrency_8"  # No config, use run name

    def test_extract_experiment_group_no_valid_parent(self, tmp_path: Path) -> None:
        """Test that runs without valid parent use run name."""
        config = ExperimentClassificationConfig(
            baselines=["*baseline*"],
            treatments=["*treatment*"],
            default="treatment",
        )
        loader = DataLoader(classification_config=config)

        # Top-level run (parent is tmp_path, which likely doesn't match patterns)
        run_path = tmp_path / "treatment_1"
        run_name = "treatment_1"

        result = loader._extract_experiment_group(run_path, run_name)
        # Parent (tmp_path) unlikely to match "*treatment*", so use run_name
        assert result == "treatment_1"

    @pytest.mark.parametrize(
        "pattern,run_name,expected",
        [
            ("*", "anything", "baseline"),  # Wildcard matches all
            ("exact_match", "exact_match", "baseline"),  # Exact match
            ("*prefix", "my_prefix", "baseline"),  # Suffix match
            ("suffix*", "suffix_test", "baseline"),  # Prefix match
            ("*middle*", "has_middle_part", "baseline"),  # Middle match
        ],
    )
    def test_classify_with_various_patterns(
        self, tmp_path: Path, pattern: str, run_name: str, expected: str
    ) -> None:
        """Test classification with various glob patterns."""
        config = ExperimentClassificationConfig(
            baselines=[pattern],
            treatments=[],
            default="treatment",
        )
        loader = DataLoader(classification_config=config)

        run_path = tmp_path / run_name
        result = loader._classify_experiment_type(run_path, run_name)
        assert result == expected

    def test_empty_pattern_lists(self, tmp_path: Path) -> None:
        """Test with empty baseline and treatment lists."""
        config = ExperimentClassificationConfig(
            baselines=[],
            treatments=[],
            default="baseline",
        )
        loader = DataLoader(classification_config=config)

        run_path = tmp_path / "test_run"
        run_name = "test_run"

        result = loader._classify_experiment_type(run_path, run_name)
        assert result == "baseline"  # Should use default

    def test_extract_experiment_group_nested_structure(self, tmp_path: Path) -> None:
        """Test that nested runs group by immediate parent if it matches."""
        config = ExperimentClassificationConfig(
            baselines=["*baseline*"],
            treatments=["*treatment*"],
            default="treatment",
        )
        loader = DataLoader(classification_config=config)

        # Create: experiments/treatment_3/concurrency_8/
        nested = tmp_path / "experiments" / "treatment_3" / "concurrency_8"
        nested.mkdir(parents=True)

        result = loader._extract_experiment_group(nested, "concurrency_8")
        # Immediate parent "treatment_3" matches "*treatment*"
        assert result == "treatment_3"

    def test_extract_experiment_group_deeply_nested_checks_immediate_parent(
        self, tmp_path: Path
    ) -> None:
        """Test that only immediate parent is checked, not ancestors."""
        config = ExperimentClassificationConfig(
            baselines=["*baseline*"],
            treatments=["*treatment*"],
            default="treatment",
        )
        loader = DataLoader(classification_config=config)

        # Create: treatment_1/artifacts/run/
        # Parent "artifacts" doesn't match patterns, so use run name
        nested = tmp_path / "treatment_1" / "artifacts" / "run"
        nested.mkdir(parents=True)

        result = loader._extract_experiment_group(nested, "run")
        # Immediate parent "artifacts" doesn't match, use run name
        assert result == "run"
