# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for swept parameter detection and auto-selection functions.

This module tests the functionality for detecting which configuration parameters
vary across runs and automatically selecting appropriate parameters for grouping
and labeling in multi-run plots.
"""

import pandas as pd
import pytest

from aiperf.plot.core.swept_params import (
    auto_select_group_by,
    auto_select_label_by,
    detect_swept_parameters,
)


@pytest.fixture
def multi_run_df():
    """
    Create sample multi-run DataFrame with varying parameters.

    Returns:
        DataFrame with model, concurrency, and other config params.
    """
    return pd.DataFrame(
        {
            "model": ["model-a"] * 3 + ["model-b"] * 3,
            "concurrency": [1, 4, 8] * 2,
            "batch_size": [32] * 6,  # Constant
            "temperature": [0.7] * 6,  # Constant
            "profiling.start_time": [1000 + i for i in range(6)],  # Should be ignored
            "endpoint.url": [f"http://host{i}" for i in range(6)],  # Should be ignored
            "metric_value": [100 + i * 10 for i in range(6)],
        }
    )


@pytest.fixture
def empty_df():
    """
    Create empty DataFrame for edge case testing.

    Returns:
        Empty DataFrame.
    """
    return pd.DataFrame()


@pytest.fixture
def constant_params_df():
    """
    Create DataFrame where all parameters are constant.

    Returns:
        DataFrame with no varying parameters.
    """
    return pd.DataFrame(
        {
            "model": ["model-a"] * 5,
            "concurrency": [4] * 5,
            "temperature": [0.7] * 5,
        }
    )


@pytest.fixture
def many_groups_df():
    """
    Create DataFrame with parameter having many unique values (>10).

    Returns:
        DataFrame with high-cardinality parameter.
    """
    return pd.DataFrame(
        {
            "model": [f"model-{i}" for i in range(15)],
            "concurrency": list(range(1, 16)),
            "batch_size": [32] * 15,
        }
    )


class TestDetectSweptParameters:
    """Tests for detect_swept_parameters function."""

    def test_detect_swept_basic(self, multi_run_df):
        """Test detection of swept parameters in typical multi-run scenario."""
        swept = detect_swept_parameters(multi_run_df)

        # Should detect model and concurrency as swept (varying)
        assert "model" in swept
        assert "concurrency" in swept

        # Should not detect constants
        assert "batch_size" not in swept
        assert "temperature" not in swept

        # Should ignore default ignore params
        assert "profiling.start_time" not in swept
        assert "endpoint.url" not in swept

    def test_detect_swept_empty_dataframe(self, empty_df):
        """Test detection with empty DataFrame returns empty list."""
        swept = detect_swept_parameters(empty_df)
        assert swept == []

    def test_detect_swept_no_varying_params(self, constant_params_df):
        """Test detection when all parameters are constant."""
        swept = detect_swept_parameters(constant_params_df)
        assert swept == []

    def test_detect_swept_custom_ignore_set(self):
        """Test detection with custom ignore parameter set."""
        df = pd.DataFrame(
            {
                "model": ["a", "b", "a"],
                "custom_param": [1, 2, 3],
                "ignore_me": [10, 20, 30],
            }
        )

        # Use custom ignore set
        swept = detect_swept_parameters(df, ignore_params={"ignore_me"})

        assert "model" in swept
        assert "custom_param" in swept
        assert "ignore_me" not in swept

    def test_detect_swept_with_nans(self):
        """Test detection handles NaN values correctly."""
        df = pd.DataFrame(
            {
                "model": ["a", "b", None, "a"],
                "concurrency": [1, 2, None, 1],
            }
        )

        swept = detect_swept_parameters(df)

        # Both should be detected as varying (NaN is dropped before uniqueness check)
        assert "model" in swept
        assert "concurrency" in swept


class TestAutoSelectGroupBy:
    """Tests for auto_select_group_by function."""

    def test_auto_select_group_by_prefers_model(self, multi_run_df):
        """Test that 'model' is preferred when it varies."""
        group_by = auto_select_group_by(multi_run_df)
        assert group_by == "model"

    def test_auto_select_group_by_no_swept_params(self, constant_params_df):
        """Test returns None when no parameters vary."""
        group_by = auto_select_group_by(constant_params_df)
        assert group_by is None

    def test_auto_select_group_by_no_model(self):
        """Test selection when model is not in swept params."""
        df = pd.DataFrame(
            {
                "concurrency": [1, 2, 4, 8],
                "batch_size": [16, 32, 16, 32],
            }
        )

        group_by = auto_select_group_by(df)

        # Should select parameter with fewer unique values
        # batch_size has 2 unique values, concurrency has 4
        assert group_by == "batch_size"

    def test_auto_select_group_by_all_too_many_values(self, many_groups_df):
        """Test fallback when all parameters have >10 unique values."""
        # Both model and concurrency have >10 unique values
        group_by = auto_select_group_by(many_groups_df)

        # Should fallback to first swept param
        swept = detect_swept_parameters(many_groups_df)
        assert group_by == swept[0]

    def test_auto_select_group_by_with_provided_swept_params(self, multi_run_df):
        """Test using provided swept_params instead of auto-detection."""
        # Provide only concurrency as swept param
        group_by = auto_select_group_by(multi_run_df, swept_params=["concurrency"])
        assert group_by == "concurrency"


class TestAutoSelectLabelBy:
    """Tests for auto_select_label_by function."""

    def test_auto_select_label_by_prefers_concurrency(self, multi_run_df):
        """Test that 'concurrency' is preferred when it varies."""
        label_by = auto_select_label_by(multi_run_df)
        assert label_by == "concurrency"

    def test_auto_select_label_by_no_swept_params(self, constant_params_df):
        """Test returns None when no parameters vary."""
        label_by = auto_select_label_by(constant_params_df)
        assert label_by is None

    def test_auto_select_label_by_avoids_group_by(self):
        """Test that label_by avoids using the same parameter as group_by."""
        df = pd.DataFrame(
            {
                "model": ["a", "b", "c"],
                "batch_size": [16, 32, 64],
                "temperature": [0.7, 0.8, 0.9],
            }
        )

        # Explicitly set group_by to "model"
        label_by = auto_select_label_by(df, group_by="model")

        # Should not be "model"
        assert label_by != "model"
        assert label_by in ["batch_size", "temperature"]

    def test_auto_select_label_by_no_concurrency(self):
        """Test selection when concurrency is not in swept params."""
        df = pd.DataFrame(
            {
                "model": ["a", "b", "c", "d"],
                "batch_size": [16, 32, 16, 32],
            }
        )

        label_by = auto_select_label_by(df)

        # Should select one of the params with 2-20 unique values
        # Both model (4) and batch_size (2) are in range
        assert label_by in ["model", "batch_size"]

    def test_auto_select_label_by_all_outside_range(self):
        """Test fallback when all params outside 2-20 unique values range."""
        df = pd.DataFrame(
            {
                "model": [f"model-{i}" for i in range(25)],
                "param_single": [1] * 25,  # Only 1 unique value
            }
        )

        label_by = auto_select_label_by(df)

        # Should fallback to first swept param
        # Only "model" varies, param_single is constant
        assert label_by == "model"

    def test_auto_select_label_by_with_provided_swept_params(self, multi_run_df):
        """Test using provided swept_params instead of auto-detection."""
        # Provide only model as swept param
        label_by = auto_select_label_by(multi_run_df, swept_params=["model"])
        assert label_by == "model"

    def test_auto_select_label_by_empty_swept_params(self):
        """Test returns None when swept_params list is explicitly empty."""
        df = pd.DataFrame({"model": ["a", "b", "c"]})
        label_by = auto_select_label_by(df, swept_params=[])
        assert label_by is None
