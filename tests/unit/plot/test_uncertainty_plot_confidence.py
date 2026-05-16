# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Property-based tests for ellipse geometry utility.

Feature: latency-throughput-uncertainty-plot
"""

import math
from pathlib import Path

import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import orjson
import pandas as pd
import plotly.graph_objects as go
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError
from scipy.stats import chi2

from aiperf.plot.config import PlotConfig
from aiperf.plot.constants import PlotTheme
from aiperf.plot.core.plot_generator import PlotGenerator
from aiperf.plot.core.plot_specs import DataSource, MetricSpec, PlotSpec
from aiperf.plot.dashboard.builder import _build_multi_run_plot_types
from aiperf.plot.dashboard.callbacks import _build_uncertainty_figure
from aiperf.plot.exporters import export_uncertainty_matplotlib
from aiperf.plot.geometry import (
    compute_axis_aligned_ellipse_vertices,
    compute_ellipse_vertices,
)
from aiperf.plot.handlers.multi_run_handlers import (
    LatencyThroughputUncertaintyHandler,
    _build_uncertainty_points,
)
from aiperf.plot.models.uncertainty import (
    BenchmarkPoint,
    LatencyThroughputUncertaintyData,
)
from aiperf.plot.renderers import render_matplotlib_uncertainty
from aiperf.plugin import plugins
from aiperf.plugin.enums import PlotType, PluginType

# --- Shared strategies ---

finite_floats = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)
positive_floats = st.floats(
    min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False
)
valid_confidence_levels = st.sampled_from([0.90, 0.95, 0.99])
vertex_counts = st.integers(min_value=8, max_value=256)


def _find_plotly_mean_trace(fig: go.Figure) -> go.Scatter | None:
    """Find the mean-point trace (has error_x and error_y set)."""
    for trace in fig.data:
        if (
            hasattr(trace, "error_x")
            and trace.error_x is not None
            and trace.error_x.array is not None
        ):
            return trace
    return None


@st.composite
def positive_semidefinite_2x2(draw: st.DrawFn) -> np.ndarray:
    """Generate a valid positive semi-definite 2x2 matrix via Cholesky decomposition."""
    a = draw(
        st.floats(
            min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False
        )
    )
    b = draw(
        st.floats(
            min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False
        )
    )
    c = draw(
        st.floats(
            min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False
        )
    )
    L = np.array([[a, 0.0], [b, c]])
    return L @ L.T


@st.composite
def positive_definite_2x2(draw: st.DrawFn) -> np.ndarray:
    """Generate a strictly positive definite 2x2 matrix via Cholesky decomposition.

    Ensures both eigenvalues are well-separated from zero for round-trip fitting.
    """
    a = draw(
        st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False)
    )
    b = draw(
        st.floats(
            min_value=-20.0, max_value=20.0, allow_nan=False, allow_infinity=False
        )
    )
    c = draw(
        st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False)
    )
    L = np.array([[a, 0.0], [b, c]])
    return L @ L.T


# --- Property 3: Ellipse polygon is closed with correct vertex count ---


class TestEllipsePolygonClosedWithCorrectVertexCount:
    """Property 3: Ellipse polygon is closed with correct vertex count.

    For any valid positive semi-definite 2x2 covariance matrix, center point,
    valid confidence level, and vertex count n, `compute_ellipse_vertices`
    SHALL return a list of length n + 1 where the first and last elements are
    equal within floating-point tolerance.

    **Validates: Requirements 2.1**
    """

    @given(
        cov=positive_semidefinite_2x2(),
        cx=finite_floats,
        cy=finite_floats,
        confidence=valid_confidence_levels,
        n=vertex_counts,
    )
    @settings(max_examples=100, deadline=None)
    def test_vertex_count_is_n_plus_one(
        self,
        cov: np.ndarray,
        cx: float,
        cy: float,
        confidence: float,
        n: int,
    ) -> None:
        """Returned list has exactly n + 1 vertices."""
        vertices = compute_ellipse_vertices(cov, (cx, cy), confidence, n_vertices=n)
        assert len(vertices) == n + 1

    @given(
        cov=positive_semidefinite_2x2(),
        cx=finite_floats,
        cy=finite_floats,
        confidence=valid_confidence_levels,
        n=vertex_counts,
    )
    @settings(max_examples=100, deadline=None)
    def test_first_and_last_vertex_are_equal(
        self,
        cov: np.ndarray,
        cx: float,
        cy: float,
        confidence: float,
        n: int,
    ) -> None:
        """First and last vertices are equal within floating-point tolerance."""
        vertices = compute_ellipse_vertices(cov, (cx, cy), confidence, n_vertices=n)
        first = vertices[0]
        last = vertices[-1]
        assert abs(first[0] - last[0]) < 1e-10, (
            f"First x={first[0]} != last x={last[0]}"
        )
        assert abs(first[1] - last[1]) < 1e-10, (
            f"First y={first[1]} != last y={last[1]}"
        )


# --- Property 4: Axis-aligned fallback produces unrotated ellipse ---


class TestAxisAlignedFallbackProducesUnrotatedEllipse:
    """Property 4: Axis-aligned fallback produces unrotated ellipse.

    For any center point and positive x/y radii,
    `compute_axis_aligned_ellipse_vertices` SHALL produce vertices that satisfy
    the axis-aligned ellipse equation ((x - cx) / rx)^2 + ((y - cy) / ry)^2
    approx 1.0 within floating-point tolerance.

    **Validates: Requirements 2.4**
    """

    @given(
        cx=st.floats(
            min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
        ),
        cy=st.floats(
            min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
        ),
        rx=st.floats(
            min_value=0.1, max_value=1e3, allow_nan=False, allow_infinity=False
        ),
        ry=st.floats(
            min_value=0.1, max_value=1e3, allow_nan=False, allow_infinity=False
        ),
        n=vertex_counts,
    )
    @settings(max_examples=100, deadline=None)
    def test_vertices_satisfy_ellipse_equation(
        self,
        cx: float,
        cy: float,
        rx: float,
        ry: float,
        n: int,
    ) -> None:
        """All non-closing vertices satisfy ((x-cx)/rx)^2 + ((y-cy)/ry)^2 approx 1."""
        vertices = compute_axis_aligned_ellipse_vertices((cx, cy), rx, ry, n_vertices=n)
        # Exclude the closing vertex (duplicate of first)
        for x, y in vertices[:-1]:
            val = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
            assert abs(val - 1.0) < 1e-8, (
                f"Ellipse equation value {val} != 1.0 at ({x}, {y})"
            )


# --- Property 5: Ellipse geometry round-trip ---


def _fit_ellipse_from_vertices(
    vertices: list[tuple[float, float]],
) -> tuple[float, float, float]:
    """Fit a general conic to vertices and extract semi-axes and rotation.

    Fits Ax^2 + Bxy + Cy^2 + Dx + Ey = 1 via least-squares, then extracts
    semi-axes and rotation angle from the conic coefficients.

    Returns:
        (semi_major, semi_minor, rotation_angle) where semi_major >= semi_minor.
        rotation_angle is the angle of the semi-major axis from the x-axis.
    """
    # Exclude closing vertex
    pts = np.array(vertices[:-1])
    x = pts[:, 0]
    y = pts[:, 1]

    # Build design matrix for Ax^2 + Bxy + Cy^2 + Dx + Ey = 1
    D = np.column_stack([x**2, x * y, y**2, x, y])
    rhs = np.ones(len(x))
    coeffs, _, _, _ = np.linalg.lstsq(D, rhs, rcond=None)
    A, B, C, D_coeff, E = coeffs

    # Center of the ellipse
    denom = 4 * A * C - B**2
    x0 = (B * E - 2 * C * D_coeff) / denom
    y0 = (B * D_coeff - 2 * A * E) / denom

    # Conic matrix
    M = np.array([[A, B / 2], [B / 2, C]])
    eig_vals, eig_vecs = np.linalg.eigh(M)

    # Value at center for scaling
    val_center = A * x0**2 + B * x0 * y0 + C * y0**2 + D_coeff * x0 + E * y0
    scale_factor = 1.0 - val_center

    # Semi-axes: smaller eigenvalue -> larger semi-axis
    semi_axes = np.sqrt(scale_factor / eig_vals)
    idx_major = np.argmax(semi_axes)
    semi_major = semi_axes[idx_major]
    semi_minor = semi_axes[1 - idx_major]

    # Rotation angle of the semi-major axis
    angle = math.atan2(eig_vecs[1, idx_major], eig_vecs[0, idx_major])

    return semi_major, semi_minor, angle


def _normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi/2, pi/2] range for comparison.

    Ellipses have pi-periodicity in rotation, so angles differing by pi
    represent the same ellipse.
    """
    angle = angle % math.pi
    if angle > math.pi / 2:
        angle -= math.pi
    return angle


class TestEllipseGeometryRoundTrip:
    """Property 5: Ellipse geometry round-trip.

    For any valid positive definite 2x2 covariance matrix, computing ellipse
    vertices via `compute_ellipse_vertices` and then fitting an ellipse back
    to those vertices SHALL recover the original semi-axes and rotation angle
    within 1e-6 tolerance.

    **Validates: Requirements 2.6**
    """

    @given(
        cov=positive_definite_2x2(),
        cx=st.floats(
            min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
        cy=st.floats(
            min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
        confidence=valid_confidence_levels,
    )
    @settings(max_examples=100, deadline=None)
    def test_round_trip_recovers_semi_axes_and_rotation(
        self,
        cov: np.ndarray,
        cx: float,
        cy: float,
        confidence: float,
    ) -> None:
        """Fitting an ellipse to generated vertices recovers original parameters."""
        n_vertices = 128  # Use more vertices for better fitting accuracy

        # Compute expected semi-axes and rotation from the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.maximum(eigenvalues, 1e-12)
        scale = math.sqrt(chi2.ppf(confidence, df=2))
        expected_a = scale * math.sqrt(float(eigenvalues[1]))
        expected_b = scale * math.sqrt(float(eigenvalues[0]))

        # expected_a corresponds to eigenvalues[1] (larger), so it's the semi-major
        expected_major = expected_a
        expected_minor = expected_b
        # Angle of the semi-major axis (eigenvector for eigenvalues[1])
        expected_theta = math.atan2(
            float(eigenvectors[1, 1]), float(eigenvectors[0, 1])
        )

        # Generate vertices
        vertices = compute_ellipse_vertices(
            cov, (cx, cy), confidence, n_vertices=n_vertices
        )

        # Fit ellipse back from vertices
        fitted_major, fitted_minor, fitted_theta = _fit_ellipse_from_vertices(vertices)

        assert abs(fitted_major - expected_major) < 1e-6, (
            f"Semi-major: fitted={fitted_major}, expected={expected_major}"
        )
        assert abs(fitted_minor - expected_minor) < 1e-6, (
            f"Semi-minor: fitted={fitted_minor}, expected={expected_minor}"
        )

        # Compare rotation angles only when the ellipse is not nearly circular.
        # When semi-axes are nearly equal, rotation is numerically undefined.
        if abs(expected_major - expected_minor) > 1e-3 * expected_major:
            norm_expected = _normalize_angle(expected_theta)
            norm_fitted = _normalize_angle(fitted_theta)
            angle_diff = abs(norm_expected - norm_fitted)
            # Handle wrap-around near pi/2 boundary
            angle_diff = min(angle_diff, math.pi - angle_diff)
            assert angle_diff < 1e-6, (
                f"Rotation: fitted={norm_fitted}, expected={norm_expected}, diff={angle_diff}"
            )


# --- Property 13: orjson serialization round-trip for ellipse vertices ---


class TestOrjsonSerializationRoundTrip:
    """Property 13: orjson serialization round-trip for ellipse vertices.

    For any valid positive semi-definite 2x2 covariance matrix and valid
    confidence level, serializing the output of `compute_ellipse_vertices`
    via `orjson.dumps` and deserializing via `orjson.loads` SHALL produce
    values equal to the originals within 1e-12 absolute tolerance.

    **Validates: Requirements 9.1, 9.2**
    """

    @given(
        cov=positive_semidefinite_2x2(),
        cx=finite_floats,
        cy=finite_floats,
        confidence=valid_confidence_levels,
    )
    @settings(max_examples=100, deadline=None)
    def test_serialization_round_trip_preserves_precision(
        self,
        cov: np.ndarray,
        cx: float,
        cy: float,
        confidence: float,
    ) -> None:
        """orjson round-trip preserves vertex coordinates within 1e-12."""
        vertices = compute_ellipse_vertices(cov, (cx, cy), confidence)

        serialized = orjson.dumps(vertices)
        deserialized = orjson.loads(serialized)

        assert len(deserialized) == len(vertices)
        for (orig_x, orig_y), restored in zip(vertices, deserialized, strict=True):
            restored_x, restored_y = restored
            assert abs(orig_x - restored_x) < 1e-12, (
                f"x mismatch: {orig_x} vs {restored_x}"
            )
            assert abs(orig_y - restored_y) < 1e-12, (
                f"y mismatch: {orig_y} vs {restored_y}"
            )


# --- Property 1: BenchmarkPoint CI ordering validation ---


class TestBenchmarkPointCIOrderingValidation:
    """Property 1: BenchmarkPoint CI ordering validation.

    For any pair of floats (ci_low, mean) where ci_low > mean, constructing a
    BenchmarkPoint with those values SHALL raise a ValidationError.
    Symmetrically, for any (mean, ci_high) where ci_high < mean, construction
    SHALL raise a ValidationError.

    **Validates: Requirements 1.4**
    """

    @given(
        x_mean=finite_floats,
        y_mean=finite_floats,
        offset=st.floats(
            min_value=1e-9, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_x_ci_low_greater_than_mean_raises(
        self,
        x_mean: float,
        y_mean: float,
        offset: float,
    ) -> None:
        """x_ci_low > x_mean SHALL raise ValidationError."""
        with pytest.raises(ValidationError):
            BenchmarkPoint(
                x_mean=x_mean,
                y_mean=y_mean,
                x_ci_low=x_mean + offset,
                x_ci_high=x_mean + offset + 1.0,
                y_ci_low=y_mean - 1.0,
                y_ci_high=y_mean + 1.0,
            )

    @given(
        x_mean=finite_floats,
        y_mean=finite_floats,
        offset=st.floats(
            min_value=1e-9, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_x_ci_high_less_than_mean_raises(
        self,
        x_mean: float,
        y_mean: float,
        offset: float,
    ) -> None:
        """x_ci_high < x_mean SHALL raise ValidationError."""
        with pytest.raises(ValidationError):
            BenchmarkPoint(
                x_mean=x_mean,
                y_mean=y_mean,
                x_ci_low=x_mean - offset - 1.0,
                x_ci_high=x_mean - offset,
                y_ci_low=y_mean - 1.0,
                y_ci_high=y_mean + 1.0,
            )

    @given(
        x_mean=finite_floats,
        y_mean=finite_floats,
        offset=st.floats(
            min_value=1e-9, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_y_ci_low_greater_than_mean_raises(
        self,
        x_mean: float,
        y_mean: float,
        offset: float,
    ) -> None:
        """y_ci_low > y_mean SHALL raise ValidationError."""
        with pytest.raises(ValidationError):
            BenchmarkPoint(
                x_mean=x_mean,
                y_mean=y_mean,
                x_ci_low=x_mean - 1.0,
                x_ci_high=x_mean + 1.0,
                y_ci_low=y_mean + offset,
                y_ci_high=y_mean + offset + 1.0,
            )

    @given(
        x_mean=finite_floats,
        y_mean=finite_floats,
        offset=st.floats(
            min_value=1e-9, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_y_ci_high_less_than_mean_raises(
        self,
        x_mean: float,
        y_mean: float,
        offset: float,
    ) -> None:
        """y_ci_high < y_mean SHALL raise ValidationError."""
        with pytest.raises(ValidationError):
            BenchmarkPoint(
                x_mean=x_mean,
                y_mean=y_mean,
                x_ci_low=x_mean - 1.0,
                x_ci_high=x_mean + 1.0,
                y_ci_low=y_mean - offset - 1.0,
                y_ci_high=y_mean - offset,
            )


# --- Property 2: Confidence level validation ---


class TestConfidenceLevelValidation:
    """Property 2: Confidence level validation.

    For any float value not in the set {0.90, 0.95, 0.99}, constructing a
    LatencyThroughputUncertaintyData with that confidence_level SHALL raise
    a ValidationError.

    **Validates: Requirements 1.3**
    """

    @given(
        confidence=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ).filter(lambda x: x not in {0.90, 0.95, 0.99}),
    )
    @settings(max_examples=100, deadline=None)
    def test_invalid_confidence_level_raises(self, confidence: float) -> None:
        """Any confidence_level not in {0.90, 0.95, 0.99} SHALL raise ValidationError."""
        with pytest.raises(ValidationError):
            LatencyThroughputUncertaintyData(
                points=[],
                confidence_level=confidence,
            )


# --- Unit tests for data contract models (Task 4.4) ---


class TestDataContractModels:
    """Unit tests for BenchmarkPoint and LatencyThroughputUncertaintyData.

    Tests valid construction, ValidationError on invalid inputs, and default values.

    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**
    """

    def test_valid_benchmark_point_construction(self) -> None:
        """Valid BenchmarkPoint with all fields set constructs without error."""
        point = BenchmarkPoint(
            x_mean=10.0,
            y_mean=100.0,
            x_ci_low=8.0,
            x_ci_high=12.0,
            y_ci_low=90.0,
            y_ci_high=110.0,
            cov_xy=5.0,
            label="concurrency=4",
        )
        assert point.x_mean == 10.0
        assert point.y_mean == 100.0
        assert point.x_ci_low == 8.0
        assert point.x_ci_high == 12.0
        assert point.y_ci_low == 90.0
        assert point.y_ci_high == 110.0
        assert point.cov_xy == 5.0
        assert point.label == "concurrency=4"

    def test_benchmark_point_cov_xy_defaults_to_none(self) -> None:
        """cov_xy defaults to None when not provided."""
        point = BenchmarkPoint(
            x_mean=5.0,
            y_mean=50.0,
            x_ci_low=4.0,
            x_ci_high=6.0,
            y_ci_low=45.0,
            y_ci_high=55.0,
        )
        assert point.cov_xy is None

    def test_benchmark_point_label_defaults_to_none(self) -> None:
        """label defaults to None when not provided."""
        point = BenchmarkPoint(
            x_mean=5.0,
            y_mean=50.0,
            x_ci_low=4.0,
            x_ci_high=6.0,
            y_ci_low=45.0,
            y_ci_high=55.0,
        )
        assert point.label is None

    def test_benchmark_point_ci_equal_to_mean_is_valid(self) -> None:
        """CI bounds equal to mean (zero-width interval) is valid."""
        point = BenchmarkPoint(
            x_mean=5.0,
            y_mean=50.0,
            x_ci_low=5.0,
            x_ci_high=5.0,
            y_ci_low=50.0,
            y_ci_high=50.0,
        )
        assert point.x_ci_low == point.x_mean
        assert point.y_ci_high == point.y_mean

    def test_benchmark_point_invalid_x_ci_low_raises(self) -> None:
        """x_ci_low > x_mean raises ValidationError."""
        with pytest.raises(ValidationError):
            BenchmarkPoint(
                x_mean=10.0,
                y_mean=100.0,
                x_ci_low=11.0,
                x_ci_high=12.0,
                y_ci_low=90.0,
                y_ci_high=110.0,
            )

    def test_benchmark_point_invalid_x_ci_high_raises(self) -> None:
        """x_ci_high < x_mean raises ValidationError."""
        with pytest.raises(ValidationError):
            BenchmarkPoint(
                x_mean=10.0,
                y_mean=100.0,
                x_ci_low=8.0,
                x_ci_high=9.0,
                y_ci_low=90.0,
                y_ci_high=110.0,
            )

    def test_benchmark_point_invalid_y_ci_low_raises(self) -> None:
        """y_ci_low > y_mean raises ValidationError."""
        with pytest.raises(ValidationError):
            BenchmarkPoint(
                x_mean=10.0,
                y_mean=100.0,
                x_ci_low=8.0,
                x_ci_high=12.0,
                y_ci_low=101.0,
                y_ci_high=110.0,
            )

    def test_benchmark_point_invalid_y_ci_high_raises(self) -> None:
        """y_ci_high < y_mean raises ValidationError."""
        with pytest.raises(ValidationError):
            BenchmarkPoint(
                x_mean=10.0,
                y_mean=100.0,
                x_ci_low=8.0,
                x_ci_high=12.0,
                y_ci_low=90.0,
                y_ci_high=99.0,
            )

    def test_valid_uncertainty_data_construction(self) -> None:
        """Valid LatencyThroughputUncertaintyData constructs without error."""
        point = BenchmarkPoint(
            x_mean=10.0,
            y_mean=100.0,
            x_ci_low=8.0,
            x_ci_high=12.0,
            y_ci_low=90.0,
            y_ci_high=110.0,
        )
        data = LatencyThroughputUncertaintyData(
            points=[point],
            confidence_level=0.95,
            title="Test Plot",
            x_label="Latency (ms)",
            y_label="Throughput (tok/s)",
        )
        assert len(data.points) == 1
        assert data.confidence_level == 0.95
        assert data.title == "Test Plot"

    def test_uncertainty_data_default_confidence_level(self) -> None:
        """confidence_level defaults to 0.95."""
        data = LatencyThroughputUncertaintyData(points=[])
        assert data.confidence_level == 0.95

    @pytest.mark.parametrize(
        "level",
        [0.90, 0.95, 0.99],
    )  # fmt: skip
    def test_uncertainty_data_valid_confidence_levels(self, level: float) -> None:
        """Valid confidence level is accepted."""
        data = LatencyThroughputUncertaintyData(points=[], confidence_level=level)
        assert data.confidence_level == level

    def test_uncertainty_data_invalid_confidence_level_raises(self) -> None:
        """Invalid confidence_level raises ValidationError."""
        with pytest.raises(ValidationError):
            LatencyThroughputUncertaintyData(points=[], confidence_level=0.80)

    def test_uncertainty_data_optional_fields_default_none(self) -> None:
        """title, x_label, y_label, group_by default to None."""
        data = LatencyThroughputUncertaintyData(points=[])
        assert data.title is None
        assert data.x_label is None
        assert data.y_label is None
        assert data.group_by is None


# --- Shared strategy for valid LatencyThroughputUncertaintyData ---


@st.composite
def valid_benchmark_point(draw: st.DrawFn) -> BenchmarkPoint:
    """Generate a valid BenchmarkPoint with correct CI ordering."""
    x_mean = draw(
        st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)
    )
    y_mean = draw(
        st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)
    )
    x_offset = draw(
        st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
    )
    y_offset = draw(
        st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
    )
    use_cov = draw(st.booleans())
    use_label = draw(st.booleans())

    cov_xy = (
        draw(
            st.floats(
                min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
            )
        )
        if use_cov
        else None
    )
    label = (
        draw(
            st.text(
                min_size=1,
                max_size=20,
                alphabet=st.characters(whitelist_categories=("L", "N")),
            )
        )
        if use_label
        else None
    )

    return BenchmarkPoint(
        x_mean=x_mean,
        y_mean=y_mean,
        x_ci_low=x_mean - x_offset,
        x_ci_high=x_mean + x_offset,
        y_ci_low=y_mean - y_offset,
        y_ci_high=y_mean + y_offset,
        cov_xy=cov_xy,
        label=label,
    )


@st.composite
def valid_uncertainty_data(
    draw: st.DrawFn, min_points: int = 1
) -> LatencyThroughputUncertaintyData:
    """Generate valid LatencyThroughputUncertaintyData with 1-5 unique-x points."""
    n = draw(st.integers(min_value=min_points, max_value=5))
    # Ensure distinct x_mean values so sorted-order assertions are deterministic
    x_means = draw(
        st.lists(
            st.floats(
                min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
            ),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    points = []
    for x_mean in x_means:
        point = draw(valid_benchmark_point())
        # Override x_mean and re-derive CI bounds to maintain ordering invariant
        x_offset = point.x_ci_high - point.x_mean
        points.append(
            BenchmarkPoint(
                x_mean=x_mean,
                y_mean=point.y_mean,
                x_ci_low=x_mean - x_offset,
                x_ci_high=x_mean + x_offset,
                y_ci_low=point.y_ci_low,
                y_ci_high=point.y_ci_high,
                cov_xy=point.cov_xy,
                label=point.label,
            )
        )
    confidence = draw(valid_confidence_levels)
    return LatencyThroughputUncertaintyData(
        points=points,
        confidence_level=confidence,
        title="Test Plot",
        x_label="Latency (ms)",
        y_label="Throughput (tok/s)",
    )


# --- Property 6: Plotly renderer produces sorted mean trace with correct error bars ---


class TestPlotlyRendererSortedMeanTraceWithErrorBars:
    """Property 6: Plotly renderer produces sorted mean trace with correct error bars.

    For any valid LatencyThroughputUncertaintyData with at least one point,
    the Plotly renderer SHALL produce a figure containing a go.Scatter trace
    whose x-values are sorted ascending and whose error_x / error_y arrays
    match the asymmetric CI bounds (ci_high - mean and mean - ci_low).

    **Validates: Requirements 3.1, 3.2**
    """

    @given(data=valid_uncertainty_data(min_points=1))
    @settings(max_examples=100, deadline=None)
    def test_mean_trace_x_values_sorted_ascending(
        self,
        data: LatencyThroughputUncertaintyData,
    ) -> None:
        """Mean trace x-values are sorted in ascending order."""
        pg = PlotGenerator()
        fig = pg.create_uncertainty_plot(data)

        mean_trace = _find_plotly_mean_trace(fig)
        assert mean_trace is not None, "No mean trace with error_x found"

        x_vals = list(mean_trace.x)
        assert x_vals == sorted(x_vals), f"x-values not sorted: {x_vals}"

    @given(data=valid_uncertainty_data(min_points=1))
    @settings(max_examples=100, deadline=None)
    def test_error_bars_match_asymmetric_ci_bounds(
        self,
        data: LatencyThroughputUncertaintyData,
    ) -> None:
        """error_x and error_y arrays match asymmetric CI bounds."""
        pg = PlotGenerator()
        fig = pg.create_uncertainty_plot(data)

        mean_trace = _find_plotly_mean_trace(fig)
        assert mean_trace is not None, "No mean trace with error_x found"

        sorted_points = sorted(data.points, key=lambda p: p.x_mean)

        expected_ex_plus = [p.x_ci_high - p.x_mean for p in sorted_points]
        expected_ex_minus = [p.x_mean - p.x_ci_low for p in sorted_points]
        expected_ey_plus = [p.y_ci_high - p.y_mean for p in sorted_points]
        expected_ey_minus = [p.y_mean - p.y_ci_low for p in sorted_points]

        actual_ex_plus = list(mean_trace.error_x.array)
        actual_ex_minus = list(mean_trace.error_x.arrayminus)
        actual_ey_plus = list(mean_trace.error_y.array)
        actual_ey_minus = list(mean_trace.error_y.arrayminus)

        for i in range(len(sorted_points)):
            assert abs(actual_ex_plus[i] - expected_ex_plus[i]) < 1e-10, (
                f"error_x plus mismatch at {i}: {actual_ex_plus[i]} vs {expected_ex_plus[i]}"
            )
            assert abs(actual_ex_minus[i] - expected_ex_minus[i]) < 1e-10, (
                f"error_x minus mismatch at {i}: {actual_ex_minus[i]} vs {expected_ex_minus[i]}"
            )
            assert abs(actual_ey_plus[i] - expected_ey_plus[i]) < 1e-10, (
                f"error_y plus mismatch at {i}: {actual_ey_plus[i]} vs {expected_ey_plus[i]}"
            )
            assert abs(actual_ey_minus[i] - expected_ey_minus[i]) < 1e-10, (
                f"error_y minus mismatch at {i}: {actual_ey_minus[i]} vs {expected_ey_minus[i]}"
            )


# --- Property 7: Plotly renderer produces one ellipse trace per point ---


class TestPlotlyRendererOneEllipseTracePerPoint:
    """Property 7: Plotly renderer produces one ellipse trace per point.

    For any valid LatencyThroughputUncertaintyData with n >= 1 points,
    the Plotly renderer SHALL produce exactly n traces with fill='toself'
    (one per point), using covariance-based vertices when cov_xy is
    non-None/non-zero and axis-aligned vertices otherwise.

    **Validates: Requirements 3.3, 3.4**
    """

    @given(data=valid_uncertainty_data(min_points=1))
    @settings(max_examples=100, deadline=None)
    def test_ellipse_trace_count_equals_point_count(
        self,
        data: LatencyThroughputUncertaintyData,
    ) -> None:
        """Number of fill='toself' traces equals number of points."""
        pg = PlotGenerator()
        fig = pg.create_uncertainty_plot(data)

        ellipse_traces = [t for t in fig.data if t.fill == "toself"]
        assert len(ellipse_traces) == len(data.points), (
            f"Expected {len(data.points)} ellipse traces, got {len(ellipse_traces)}"
        )

    @given(data=valid_uncertainty_data(min_points=1))
    @settings(max_examples=100, deadline=None)
    def test_ellipse_traces_have_showlegend_false(
        self,
        data: LatencyThroughputUncertaintyData,
    ) -> None:
        """All ellipse traces have showlegend=False."""
        pg = PlotGenerator()
        fig = pg.create_uncertainty_plot(data)

        ellipse_traces = [t for t in fig.data if t.fill == "toself"]
        for i, trace in enumerate(ellipse_traces):
            assert trace.showlegend is False, (
                f"Ellipse trace {i} has showlegend={trace.showlegend}"
            )


# --- Property 8: Plotly renderer produces exactly one ellipse legend entry ---


class TestPlotlyRendererExactlyOneEllipseLegendEntry:
    """Property 8: Plotly renderer produces exactly one ellipse legend entry.

    For any valid LatencyThroughputUncertaintyData with at least one point,
    the Plotly renderer SHALL produce exactly one trace whose name contains
    "Confidence Region" and showlegend=True.

    **Validates: Requirements 3.5**
    """

    @given(data=valid_uncertainty_data(min_points=1))
    @settings(max_examples=100, deadline=None)
    def test_exactly_one_confidence_region_legend_entry(
        self,
        data: LatencyThroughputUncertaintyData,
    ) -> None:
        """Exactly one trace has name containing 'Confidence Region' and showlegend=True."""
        pg = PlotGenerator()
        fig = pg.create_uncertainty_plot(data)

        legend_traces = [
            t
            for t in fig.data
            if t.showlegend is True
            and t.name is not None
            and "Confidence Region" in t.name
        ]
        assert len(legend_traces) == 1, (
            f"Expected 1 legend entry with 'Confidence Region', got {len(legend_traces)}"
        )

    @given(data=valid_uncertainty_data(min_points=1))
    @settings(max_examples=100, deadline=None)
    def test_legend_entry_contains_confidence_level_percentage(
        self,
        data: LatencyThroughputUncertaintyData,
    ) -> None:
        """Legend entry name contains the confidence level as a percentage."""
        pg = PlotGenerator()
        fig = pg.create_uncertainty_plot(data)

        level_pct = str(int(data.confidence_level * 100))
        legend_traces = [
            t
            for t in fig.data
            if t.showlegend is True
            and t.name is not None
            and "Confidence Region" in t.name
        ]
        assert len(legend_traces) == 1
        assert level_pct in legend_traces[0].name, (
            f"Expected '{level_pct}' in legend name, got '{legend_traces[0].name}'"
        )


# --- Property 9: Plotly renderer includes text annotations only for labeled points ---


class TestPlotlyRendererTextAnnotationsForLabeledPoints:
    """Property 9: Plotly renderer includes text annotations only for labeled points.

    For any valid LatencyThroughputUncertaintyData, the number of non-empty
    text annotations on the mean-point trace SHALL equal the number of
    BenchmarkPoint entries where label is not None.

    **Validates: Requirements 3.7**
    """

    @given(data=valid_uncertainty_data(min_points=1))
    @settings(max_examples=100, deadline=None)
    def test_non_empty_text_count_equals_labeled_point_count(
        self,
        data: LatencyThroughputUncertaintyData,
    ) -> None:
        """Non-empty text annotations count matches labeled points count."""
        pg = PlotGenerator()
        fig = pg.create_uncertainty_plot(data)

        mean_trace = _find_plotly_mean_trace(fig)
        assert mean_trace is not None, "No mean trace with error_x found"

        expected_labeled_count = sum(1 for p in data.points if p.label is not None)

        if mean_trace.text is not None:
            text_list = list(mean_trace.text)
            actual_non_empty = sum(1 for t in text_list if t is not None and t != "")
        else:
            actual_non_empty = 0

        assert actual_non_empty == expected_labeled_count, (
            f"Expected {expected_labeled_count} non-empty text annotations, got {actual_non_empty}"
        )


# --- Integration test: Kaleido PNG export produces valid PNG (Task 7.1) ---


class TestKaleidoPNGExport:
    """Integration test: Kaleido PNG export produces valid PNG.

    Validates: Requirements 3.8, 8.4
    """

    def test_kaleido_export_produces_valid_png(self, tmp_path: Path) -> None:
        """Exporting uncertainty plot via Kaleido produces valid PNG file."""
        points = [
            BenchmarkPoint(
                x_mean=10.0,
                y_mean=100.0,
                x_ci_low=8.0,
                x_ci_high=12.0,
                y_ci_low=90.0,
                y_ci_high=110.0,
                cov_xy=5.0,
            ),
            BenchmarkPoint(
                x_mean=20.0,
                y_mean=200.0,
                x_ci_low=18.0,
                x_ci_high=22.0,
                y_ci_low=180.0,
                y_ci_high=220.0,
                cov_xy=None,
            ),
            BenchmarkPoint(
                x_mean=30.0,
                y_mean=150.0,
                x_ci_low=25.0,
                x_ci_high=35.0,
                y_ci_low=130.0,
                y_ci_high=170.0,
                cov_xy=10.0,
            ),
            BenchmarkPoint(
                x_mean=40.0,
                y_mean=250.0,
                x_ci_low=35.0,
                x_ci_high=45.0,
                y_ci_low=230.0,
                y_ci_high=270.0,
                cov_xy=0.0,
            ),
        ]
        data = LatencyThroughputUncertaintyData(points=points, confidence_level=0.95)

        pg = PlotGenerator()
        fig = pg.create_uncertainty_plot(data)

        output_path = tmp_path / "test.png"
        fig.write_image(str(output_path))

        assert output_path.exists()
        with open(output_path, "rb") as f:
            magic = f.read(8)
        assert magic == b"\x89PNG\r\n\x1a\n"


# --- Property 10: Matplotlib renderer produces sorted mean line with correct error bars ---


class TestMatplotlibRendererSortedMeanLineWithErrorBars:
    """Property 10: Matplotlib renderer produces sorted mean line with correct error bars.

    For any valid LatencyThroughputUncertaintyData with at least one point,
    the Matplotlib renderer SHALL produce a figure whose line data x-values
    are sorted ascending and whose errorbar containers have asymmetric error
    values matching the CI bounds.

    **Validates: Requirements 4.1, 4.2**
    """

    @given(data=valid_uncertainty_data(min_points=1))
    @settings(max_examples=100, deadline=None)
    def test_mean_line_x_values_sorted_ascending(
        self,
        data: LatencyThroughputUncertaintyData,
    ) -> None:
        """Mean line x-values are sorted in ascending order."""
        fig = render_matplotlib_uncertainty(data)
        ax = fig.axes[0]

        lines = ax.get_lines()
        assert len(lines) >= 1, "No lines found on axes"

        mean_line = lines[0]
        x_vals = list(mean_line.get_xdata())
        assert x_vals == sorted(x_vals), f"x-values not sorted: {x_vals}"

        plt.close(fig)

    @given(data=valid_uncertainty_data(min_points=1))
    @settings(max_examples=100, deadline=None)
    def test_errorbar_asymmetric_values_match_ci_bounds(
        self,
        data: LatencyThroughputUncertaintyData,
    ) -> None:
        """Errorbar containers have asymmetric error values matching CI bounds."""
        fig = render_matplotlib_uncertainty(data)
        ax = fig.axes[0]

        containers = ax.containers
        assert len(containers) >= 1, "No errorbar containers found"

        errorbar_container = containers[0]
        sorted_points = sorted(data.points, key=lambda p: p.x_mean)

        # ErrorbarContainer has lines: (data_line, caplines, barlinecols)
        # barlinecols[0] = x error bar segments, barlinecols[1] = y error bar segments
        barlinecols = errorbar_container.lines[2]

        # X error bars
        x_segments = barlinecols[0].get_segments()
        for i, seg in enumerate(x_segments):
            x_low = seg[0][0]
            x_high = seg[1][0]
            expected_low = sorted_points[i].x_ci_low
            expected_high = sorted_points[i].x_ci_high
            assert abs(x_low - expected_low) < 1e-10, (
                f"x errorbar low mismatch at {i}: {x_low} vs {expected_low}"
            )
            assert abs(x_high - expected_high) < 1e-10, (
                f"x errorbar high mismatch at {i}: {x_high} vs {expected_high}"
            )

        # Y error bars
        y_segments = barlinecols[1].get_segments()
        for i, seg in enumerate(y_segments):
            y_low = seg[0][1]
            y_high = seg[1][1]
            expected_low = sorted_points[i].y_ci_low
            expected_high = sorted_points[i].y_ci_high
            assert abs(y_low - expected_low) < 1e-10, (
                f"y errorbar low mismatch at {i}: {y_low} vs {expected_low}"
            )
            assert abs(y_high - expected_high) < 1e-10, (
                f"y errorbar high mismatch at {i}: {y_high} vs {expected_high}"
            )

        plt.close(fig)

    @given(data=valid_uncertainty_data(min_points=1))
    @settings(max_examples=100, deadline=None)
    def test_mean_line_y_values_match_sorted_points(
        self,
        data: LatencyThroughputUncertaintyData,
    ) -> None:
        """Mean line y-values correspond to sorted points' y_mean values."""
        fig = render_matplotlib_uncertainty(data)
        ax = fig.axes[0]

        mean_line = ax.get_lines()[0]
        y_vals = list(mean_line.get_ydata())
        sorted_points = sorted(data.points, key=lambda p: p.x_mean)
        expected_y = [p.y_mean for p in sorted_points]

        for i in range(len(sorted_points)):
            assert abs(y_vals[i] - expected_y[i]) < 1e-10, (
                f"y mismatch at {i}: {y_vals[i]} vs {expected_y[i]}"
            )

        plt.close(fig)


# --- Property 11: Matplotlib renderer produces one ellipse patch per point ---


class TestMatplotlibRendererOneEllipsePatchPerPoint:
    """Property 11: Matplotlib renderer produces one ellipse patch per point.

    For any valid LatencyThroughputUncertaintyData with n >= 1 points,
    the Matplotlib renderer SHALL produce exactly n Ellipse patches,
    using rotation when cov_xy is non-None/non-zero and zero rotation
    otherwise.

    **Validates: Requirements 4.3, 4.4**
    """

    @given(data=valid_uncertainty_data(min_points=1))
    @settings(max_examples=100, deadline=None)
    def test_ellipse_patch_count_equals_point_count(
        self,
        data: LatencyThroughputUncertaintyData,
    ) -> None:
        """Number of Ellipse patches equals number of points."""
        fig = render_matplotlib_uncertainty(data)
        ax = fig.axes[0]

        ellipse_patches = [
            p for p in ax.patches if isinstance(p, matplotlib.patches.Ellipse)
        ]
        assert len(ellipse_patches) == len(data.points), (
            f"Expected {len(data.points)} Ellipse patches, got {len(ellipse_patches)}"
        )

        plt.close(fig)

    @given(data=valid_uncertainty_data(min_points=1))
    @settings(max_examples=100, deadline=None)
    def test_ellipse_rotation_matches_cov_xy(
        self,
        data: LatencyThroughputUncertaintyData,
    ) -> None:
        """Ellipses use rotation when cov_xy is non-None/non-zero, zero otherwise."""
        fig = render_matplotlib_uncertainty(data)
        ax = fig.axes[0]

        ellipse_patches = [
            p for p in ax.patches if isinstance(p, matplotlib.patches.Ellipse)
        ]

        sorted_points = sorted(data.points, key=lambda p: p.x_mean)

        for i, (patch, point) in enumerate(
            zip(ellipse_patches, sorted_points, strict=True)
        ):
            if point.cov_xy is None or point.cov_xy == 0:
                assert math.isclose(patch.angle, 0.0, abs_tol=1e-6), (
                    f"Ellipse {i} has angle={patch.angle} but cov_xy is None/zero"
                )
            # Non-zero cov_xy may or may not produce visible rotation depending
            # on the variance ratio, so we only assert the zero-angle case.

        plt.close(fig)

    @given(data=valid_uncertainty_data(min_points=1))
    @settings(max_examples=100, deadline=None)
    def test_ellipse_centers_match_point_means(
        self,
        data: LatencyThroughputUncertaintyData,
    ) -> None:
        """Ellipse centers match the (x_mean, y_mean) of each point."""
        fig = render_matplotlib_uncertainty(data)
        ax = fig.axes[0]

        ellipse_patches = [
            p for p in ax.patches if isinstance(p, matplotlib.patches.Ellipse)
        ]

        sorted_points = sorted(data.points, key=lambda p: p.x_mean)

        for i, (patch, point) in enumerate(
            zip(ellipse_patches, sorted_points, strict=True)
        ):
            cx, cy = patch.center
            assert abs(cx - point.x_mean) < 1e-10, (
                f"Ellipse {i} center x={cx} != point x_mean={point.x_mean}"
            )
            assert abs(cy - point.y_mean) < 1e-10, (
                f"Ellipse {i} center y={cy} != point y_mean={point.y_mean}"
            )

        plt.close(fig)


# --- Property 12: Cross-backend element count consistency ---


class TestCrossBackendElementCountConsistency:
    """Property 12: Cross-backend element count consistency.

    For the same LatencyThroughputUncertaintyData, the Plotly and Matplotlib
    renderers SHALL produce the same number of mean points, error bar pairs,
    and ellipses. Both SHALL handle mixed cov_xy (some None, some not) and
    all-None cases identically.

    **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 8.1**
    """

    @given(data=valid_uncertainty_data(min_points=1))
    @settings(max_examples=100, deadline=None)
    def test_same_number_of_mean_points(
        self,
        data: LatencyThroughputUncertaintyData,
    ) -> None:
        """Both backends produce the same number of mean points."""
        pg = PlotGenerator()
        plotly_fig = pg.create_uncertainty_plot(data)
        mpl_fig = render_matplotlib_uncertainty(data)

        # Plotly: mean trace x-values count
        plotly_mean_trace = _find_plotly_mean_trace(plotly_fig)
        assert plotly_mean_trace is not None
        plotly_n = len(plotly_mean_trace.x)

        # Matplotlib: mean line x-values count
        mpl_ax = mpl_fig.axes[0]
        mpl_line = mpl_ax.get_lines()[0]
        mpl_n = len(mpl_line.get_xdata())

        assert plotly_n == mpl_n, (
            f"Mean point count mismatch: Plotly={plotly_n}, Matplotlib={mpl_n}"
        )

        plt.close(mpl_fig)

    @given(data=valid_uncertainty_data(min_points=1))
    @settings(max_examples=100, deadline=None)
    def test_same_number_of_ellipses(
        self,
        data: LatencyThroughputUncertaintyData,
    ) -> None:
        """Both backends produce the same number of ellipses."""
        pg = PlotGenerator()
        plotly_fig = pg.create_uncertainty_plot(data)
        mpl_fig = render_matplotlib_uncertainty(data)

        # Plotly: fill='toself' traces
        plotly_ellipses = [t for t in plotly_fig.data if t.fill == "toself"]

        # Matplotlib: Ellipse patches
        mpl_ax = mpl_fig.axes[0]
        mpl_ellipses = [
            p for p in mpl_ax.patches if isinstance(p, matplotlib.patches.Ellipse)
        ]

        assert len(plotly_ellipses) == len(mpl_ellipses), (
            f"Ellipse count mismatch: Plotly={len(plotly_ellipses)}, Matplotlib={len(mpl_ellipses)}"
        )

        plt.close(mpl_fig)

    @given(data=valid_uncertainty_data(min_points=1))
    @settings(max_examples=100, deadline=None)
    def test_same_number_of_error_bar_pairs(
        self,
        data: LatencyThroughputUncertaintyData,
    ) -> None:
        """Both backends produce the same number of error bar pairs."""
        pg = PlotGenerator()
        plotly_fig = pg.create_uncertainty_plot(data)
        mpl_fig = render_matplotlib_uncertainty(data)

        # Plotly: error bar count from mean trace
        plotly_mean_trace = _find_plotly_mean_trace(plotly_fig)
        assert plotly_mean_trace is not None
        plotly_err_count = len(plotly_mean_trace.error_x.array)

        # Matplotlib: error bar segment count
        mpl_ax = mpl_fig.axes[0]
        containers = mpl_ax.containers
        assert len(containers) >= 1
        barlinecols = containers[0].lines[2]
        mpl_err_count = len(barlinecols[0].get_segments())

        assert plotly_err_count == mpl_err_count, (
            f"Error bar pair count mismatch: Plotly={plotly_err_count}, Matplotlib={mpl_err_count}"
        )

        plt.close(mpl_fig)

    @given(data=valid_uncertainty_data(min_points=1))
    @settings(max_examples=100, deadline=None)
    def test_mixed_cov_xy_handled_identically(
        self,
        data: LatencyThroughputUncertaintyData,
    ) -> None:
        """Mixed cov_xy (some None, some not) produces consistent element counts."""
        pg = PlotGenerator()
        plotly_fig = pg.create_uncertainty_plot(data)
        mpl_fig = render_matplotlib_uncertainty(data)

        plotly_ellipses = [t for t in plotly_fig.data if t.fill == "toself"]
        mpl_ax = mpl_fig.axes[0]
        mpl_ellipses = [
            p for p in mpl_ax.patches if isinstance(p, matplotlib.patches.Ellipse)
        ]

        # Both should have exactly n ellipses regardless of cov_xy mix
        assert len(plotly_ellipses) == len(data.points)
        assert len(mpl_ellipses) == len(data.points)

        plt.close(mpl_fig)


# --- Unit tests for plugin registration and config parsing (Task 10.5) ---


class TestPluginRegistrationAndConfigParsing:
    """Unit tests for plugin registration and YAML config parsing.

    **Validates: Requirements 5.1, 5.2, 5.3, 5.4**
    """

    def test_plugin_registration_returns_handler_class(self) -> None:
        """plugins.get_class returns LatencyThroughputUncertaintyHandler."""
        cls = plugins.get_class(
            PluginType.PLOT, PlotType.LATENCY_THROUGHPUT_UNCERTAINTY
        )
        assert cls is LatencyThroughputUncertaintyHandler

    def test_plugin_metadata_display_name(self) -> None:
        """Plugin metadata has correct display_name."""
        meta = plugins.get_plot_metadata(PlotType.LATENCY_THROUGHPUT_UNCERTAINTY)
        assert meta.display_name == "Uncertainty Scatter"

    def test_plugin_metadata_category(self) -> None:
        """Plugin metadata has correct category."""
        meta = plugins.get_plot_metadata(PlotType.LATENCY_THROUGHPUT_UNCERTAINTY)
        assert meta.category == "comparison"

    def test_plugin_entry_description(self) -> None:
        """Plugin entry has a description."""
        entry = plugins.get_entry("plot", PlotType.LATENCY_THROUGHPUT_UNCERTAINTY)
        assert (
            "crosshair" in entry.description.lower()
            or "ellipse" in entry.description.lower()
        )

    def test_yaml_preset_parsing_with_ci_level(self, tmp_path: Path) -> None:
        """YAML preset with ci_level produces valid PlotSpec."""
        config_content = """\
visualization:
  multi_run_defaults:
    - latency_throughput_uncertainty
  multi_run_plots:
    latency_throughput_uncertainty:
      type: latency_throughput_uncertainty
      description: "Test uncertainty plot"
      x:
        metric: request_latency
        stat: avg
      y:
        metric: output_token_throughput_per_gpu
        stat: avg
      labels: [concurrency]
      groups: [model]
      ci_level: 0.95
      title: "Test Title"
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)

        config = PlotConfig(config_path=config_file)
        specs = config.get_multi_run_plot_specs()

        assert len(specs) == 1
        spec = specs[0]
        assert spec.plot_type == PlotType.LATENCY_THROUGHPUT_UNCERTAINTY
        assert spec.name == "latency_throughput_uncertainty"
        assert spec.title == "Test Title"
        # ci_level passed through via extra="allow" on AIPerfBaseModel
        assert getattr(spec, "ci_level", None) == 0.95

    def test_yaml_preset_parsing_without_ci_level(self, tmp_path: Path) -> None:
        """YAML preset without ci_level still produces valid PlotSpec."""
        config_content = """\
visualization:
  multi_run_defaults:
    - latency_throughput_uncertainty
  multi_run_plots:
    latency_throughput_uncertainty:
      type: latency_throughput_uncertainty
      description: "Test uncertainty plot"
      x:
        metric: request_latency
        stat: avg
      y:
        metric: output_token_throughput_per_gpu
        stat: avg
      labels: [concurrency]
      groups: [model]
      title: "Test Title"
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)

        config = PlotConfig(config_path=config_file)
        specs = config.get_multi_run_plot_specs()

        assert len(specs) == 1
        spec = specs[0]
        assert spec.plot_type == PlotType.LATENCY_THROUGHPUT_UNCERTAINTY
        # ci_level not set, handler should default to 0.95
        assert getattr(spec, "ci_level", 0.95) == 0.95


# --- Unit test for dashboard dropdown inclusion (Task 11.3) ---


class TestDashboardDropdownInclusion:
    """Unit test for dashboard dropdown inclusion.

    **Validates: Requirements 6.1**
    """

    def test_build_multi_run_plot_types_includes_uncertainty(self) -> None:
        """_build_multi_run_plot_types returns entry with LATENCY_THROUGHPUT_UNCERTAINTY."""
        plot_types = _build_multi_run_plot_types()
        values = [entry["value"] for entry in plot_types]
        assert PlotType.LATENCY_THROUGHPUT_UNCERTAINTY in values

    def test_build_multi_run_plot_types_uncertainty_label(self) -> None:
        """Uncertainty entry has correct label from plugin metadata."""
        plot_types = _build_multi_run_plot_types()
        uncertainty_entry = next(
            e
            for e in plot_types
            if e["value"] == PlotType.LATENCY_THROUGHPUT_UNCERTAINTY
        )
        assert uncertainty_entry["label"] == "Uncertainty Scatter"


# --- Unit test for Matplotlib code-gen export integration (Task 12.1) ---


class TestExportUncertaintyMatplotlib:
    """Unit tests for the Matplotlib code-gen reporting entry point.

    **Validates: Requirements 4.7, 7.1**
    """

    def test_export_produces_valid_png(self, tmp_path: Path) -> None:
        """export_uncertainty_matplotlib writes a valid PNG file."""
        data = LatencyThroughputUncertaintyData(
            points=[
                BenchmarkPoint(
                    x_mean=10.0,
                    y_mean=100.0,
                    x_ci_low=8.0,
                    x_ci_high=12.0,
                    y_ci_low=90.0,
                    y_ci_high=110.0,
                    cov_xy=5.0,
                    label="c=1",
                ),
                BenchmarkPoint(
                    x_mean=20.0,
                    y_mean=200.0,
                    x_ci_low=18.0,
                    x_ci_high=22.0,
                    y_ci_low=180.0,
                    y_ci_high=220.0,
                ),
            ],
            confidence_level=0.95,
            title="Test Export",
        )

        out = tmp_path / "uncertainty.png"
        result = export_uncertainty_matplotlib(data, out)

        assert result == out
        assert out.exists()
        magic = out.read_bytes()[:8]
        assert magic == b"\x89PNG\r\n\x1a\n"

    def test_export_creates_parent_directories(self, tmp_path: Path) -> None:
        """export_uncertainty_matplotlib creates missing parent dirs."""
        data = LatencyThroughputUncertaintyData(
            points=[
                BenchmarkPoint(
                    x_mean=5.0,
                    y_mean=50.0,
                    x_ci_low=4.0,
                    x_ci_high=6.0,
                    y_ci_low=45.0,
                    y_ci_high=55.0,
                ),
            ],
            confidence_level=0.95,
        )

        nested = tmp_path / "sub" / "dir" / "plot.png"
        result = export_uncertainty_matplotlib(data, nested)

        assert result == nested
        assert nested.exists()

    def test_export_empty_data_produces_valid_png(self, tmp_path: Path) -> None:
        """Empty points list still produces a valid PNG."""
        data = LatencyThroughputUncertaintyData(
            points=[],
            confidence_level=0.95,
            title="Empty",
        )

        out = tmp_path / "empty.png"
        result = export_uncertainty_matplotlib(data, out)

        assert result == out
        assert out.exists()
        magic = out.read_bytes()[:8]
        assert magic == b"\x89PNG\r\n\x1a\n"

    @pytest.mark.parametrize(
        "theme",
        [PlotTheme.LIGHT, PlotTheme.DARK],
    )  # fmt: skip
    def test_export_respects_theme(self, tmp_path: Path, theme: PlotTheme) -> None:
        """Both light and dark themes produce valid PNGs."""
        data = LatencyThroughputUncertaintyData(
            points=[
                BenchmarkPoint(
                    x_mean=10.0,
                    y_mean=100.0,
                    x_ci_low=8.0,
                    x_ci_high=12.0,
                    y_ci_low=90.0,
                    y_ci_high=110.0,
                ),
            ],
            confidence_level=0.95,
        )

        out = tmp_path / f"{theme.value}.png"
        export_uncertainty_matplotlib(data, out, theme=theme)
        assert out.exists()
        assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# Additional unit tests for patch coverage
# ---------------------------------------------------------------------------

# --- Tests for _build_uncertainty_points (multi_run_handlers.py) ---


class TestBuildUncertaintyPoints:
    """Unit tests for _build_uncertainty_points covering grouped DataFrames."""

    def test_multiple_groups_multiple_rows(self) -> None:
        """Multiple concurrency groups with multiple rows each produce correct points."""
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
                "y": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
                "concurrency": [1, 1, 1, 4, 4, 4],
            }
        )
        points = _build_uncertainty_points(
            df, "x", "y", group_col="concurrency", label_col=None, ci_level=0.95
        )
        assert len(points) == 2
        assert points[0].n_runs == 3
        assert points[1].n_runs == 3
        assert points[0].x_mean == pytest.approx(2.0)
        assert points[1].x_mean == pytest.approx(20.0)
        assert points[0].x_ci_low < points[0].x_mean
        assert points[0].x_ci_high > points[0].x_mean

    def test_single_row_group_ci_is_zero(self) -> None:
        """A group with n=1 produces zero-width CI (ci_half = 0)."""
        df = pd.DataFrame({"x": [5.0], "y": [50.0], "concurrency": [1]})
        points = _build_uncertainty_points(
            df, "x", "y", group_col="concurrency", label_col=None, ci_level=0.95
        )
        assert len(points) == 1
        p = points[0]
        assert p.n_runs == 1
        assert p.x_ci_low == p.x_mean
        assert p.x_ci_high == p.x_mean
        assert p.y_ci_low == p.y_mean
        assert p.y_ci_high == p.y_mean

    def test_list_valued_group_col_normalizes(self) -> None:
        """A list-valued group_col picks the first matching column."""
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0],
                "y": [10.0, 20.0],
                "concurrency": [1, 2],
            }
        )
        points = _build_uncertainty_points(
            df,
            "x",
            "y",
            group_col=["missing_col", "concurrency"],
            label_col=None,
            ci_level=0.95,
        )
        assert len(points) == 2

    def test_nan_values_in_group_col_dropped(self) -> None:
        """NaN values in the group column are dropped from grouping."""
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0],
                "y": [10.0, 20.0, 30.0],
                "concurrency": [1.0, float("nan"), 2.0],
            }
        )
        points = _build_uncertainty_points(
            df, "x", "y", group_col="concurrency", label_col=None, ci_level=0.95
        )
        assert len(points) == 2
        assert all(p.n_runs == 1 for p in points)

    def test_label_col_all_nan_gives_none(self) -> None:
        """When label_col values are all NaN, label_val is None."""
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0],
                "y": [10.0, 20.0],
                "concurrency": [1, 1],
                "label": [float("nan"), float("nan")],
            }
        )
        points = _build_uncertainty_points(
            df, "x", "y", group_col="concurrency", label_col="label", ci_level=0.95
        )
        assert len(points) == 1
        assert points[0].label is None


# --- Tests for ellipse input validation (ellipse.py) ---


class TestEllipseInputValidation:
    """Unit tests for input validation in compute_ellipse_vertices and axis-aligned variant."""

    def test_compute_ellipse_vertices_n_vertices_lt_3_raises(self) -> None:
        """compute_ellipse_vertices with n_vertices < 3 raises ValueError."""
        cov = np.array([[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="n_vertices must be >= 3"):
            compute_ellipse_vertices(cov, (0.0, 0.0), 0.95, n_vertices=2)

    def test_compute_ellipse_vertices_confidence_below_zero_raises(self) -> None:
        """compute_ellipse_vertices with confidence_level <= 0 raises ValueError."""
        cov = np.array([[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="confidence_level must be in"):
            compute_ellipse_vertices(cov, (0.0, 0.0), 0.0, n_vertices=64)

    def test_compute_ellipse_vertices_confidence_above_one_raises(self) -> None:
        """compute_ellipse_vertices with confidence_level >= 1 raises ValueError."""
        cov = np.array([[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="confidence_level must be in"):
            compute_ellipse_vertices(cov, (0.0, 0.0), 1.0, n_vertices=64)

    def test_axis_aligned_n_vertices_lt_3_raises(self) -> None:
        """compute_axis_aligned_ellipse_vertices with n_vertices < 3 raises ValueError."""
        with pytest.raises(ValueError, match="n_vertices must be >= 3"):
            compute_axis_aligned_ellipse_vertices((0.0, 0.0), 1.0, 1.0, n_vertices=1)

    def test_negative_eigenvalues_clamped_and_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Negative eigenvalues are clamped and a warning is logged."""
        import logging

        cov = np.array([[1.0, 2.0], [2.0, 1.0]])
        with caplog.at_level(logging.WARNING, logger="aiperf.plot.geometry.ellipse"):
            vertices = compute_ellipse_vertices(cov, (0.0, 0.0), 0.95, n_vertices=8)
        assert len(vertices) == 9  # 8 + 1 closing vertex
        assert any("Non-positive-definite" in r.message for r in caplog.records)


# --- Tests for _build_uncertainty_figure (callbacks.py) ---


class TestBuildUncertaintyFigure:
    """Unit tests for _build_uncertainty_figure in the dashboard callbacks."""

    def test_returns_go_figure(self) -> None:
        """_build_uncertainty_figure returns a go.Figure."""
        df = pd.DataFrame(
            {
                "x_metric": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
                "y_metric": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
                "concurrency": [1, 1, 1, 4, 4, 4],
            }
        )
        pg = PlotGenerator()
        fig = _build_uncertainty_figure(
            df,
            "x_metric",
            "y_metric",
            pg,
            actual_group_by="concurrency",
            actual_label_by=None,
            plot_config_dict={},
            title="Test",
            x_label="X",
            y_label="Y",
        )
        assert isinstance(fig, go.Figure)

    def test_invalid_ci_level_defaults_to_095(self) -> None:
        """ci_level not in {0.90, 0.95, 0.99} defaults to 0.95."""
        df = pd.DataFrame(
            {
                "x_metric": [1.0, 2.0, 3.0],
                "y_metric": [10.0, 20.0, 30.0],
                "concurrency": [1, 1, 1],
            }
        )
        pg = PlotGenerator()
        fig = _build_uncertainty_figure(
            df,
            "x_metric",
            "y_metric",
            pg,
            actual_group_by="concurrency",
            actual_label_by=None,
            plot_config_dict={"ci_level": 0.80},
            title="Test",
            x_label="X",
            y_label="Y",
        )
        assert isinstance(fig, go.Figure)
        # Verify the fallback produced a 95% legend entry (not 80%)
        legend_names = [t.name for t in fig.data if t.name and "Confidence" in t.name]
        assert any("95%" in name for name in legend_names), (
            f"Expected 95% fallback, got legend names: {legend_names}"
        )


# --- Tests for _build_ellipse_trace covariance path (plot_generator.py) ---


class TestBuildEllipseTraceCovariancePath:
    """Unit tests for _build_ellipse_trace with non-zero cov_xy."""

    def test_covariance_path_produces_different_geometry_than_axis_aligned(
        self,
    ) -> None:
        """A BenchmarkPoint with non-zero cov_xy produces a rotated ellipse distinct from axis-aligned."""
        base_kwargs = {
            "x_mean": 10.0,
            "y_mean": 100.0,
            "x_ci_low": 8.0,
            "x_ci_high": 12.0,
            "y_ci_low": 90.0,
            "y_ci_high": 110.0,
        }
        rotated_point = BenchmarkPoint(**base_kwargs, cov_xy=1.5)
        aligned_point = BenchmarkPoint(**base_kwargs, cov_xy=None)

        pg = PlotGenerator()
        rotated_fig = pg.create_uncertainty_plot(
            LatencyThroughputUncertaintyData(
                points=[rotated_point], confidence_level=0.95
            )
        )
        aligned_fig = pg.create_uncertainty_plot(
            LatencyThroughputUncertaintyData(
                points=[aligned_point], confidence_level=0.95
            )
        )

        rotated_traces = [t for t in rotated_fig.data if t.fill == "toself"]
        aligned_traces = [t for t in aligned_fig.data if t.fill == "toself"]
        assert len(rotated_traces) == 1
        assert len(aligned_traces) == 1
        assert len(rotated_traces[0].x) == 65

        # Vertices should differ — covariance rotates the ellipse
        rotated_xs = list(rotated_traces[0].x)
        aligned_xs = list(aligned_traces[0].x)
        assert rotated_xs != aligned_xs, (
            "Covariance ellipse should differ from axis-aligned"
        )


# --- Tests for matplotlib low-n annotation path (matplotlib_uncertainty.py) ---


class TestMatplotlibLowNAnnotation:
    """Unit tests for the low-n dashed ellipse styling in the matplotlib renderer."""

    def test_low_n_ellipse_has_dashed_linestyle(self) -> None:
        """A point with n_runs=2 produces a dashed ellipse."""
        data = LatencyThroughputUncertaintyData(
            points=[
                BenchmarkPoint(
                    x_mean=10.0,
                    y_mean=100.0,
                    x_ci_low=8.0,
                    x_ci_high=12.0,
                    y_ci_low=90.0,
                    y_ci_high=110.0,
                    n_runs=2,
                ),
            ],
            confidence_level=0.95,
        )
        fig = render_matplotlib_uncertainty(data)
        ax = fig.axes[0]
        ellipses = [p for p in ax.patches if isinstance(p, matplotlib.patches.Ellipse)]
        assert len(ellipses) == 1
        assert ellipses[0].get_linestyle() == "--"
        assert ellipses[0].get_alpha() == pytest.approx(0.08)
        plt.close(fig)

    def test_normal_n_ellipse_has_solid_linestyle(self) -> None:
        """A point with n_runs=5 produces a solid ellipse."""
        data = LatencyThroughputUncertaintyData(
            points=[
                BenchmarkPoint(
                    x_mean=10.0,
                    y_mean=100.0,
                    x_ci_low=8.0,
                    x_ci_high=12.0,
                    y_ci_low=90.0,
                    y_ci_high=110.0,
                    n_runs=5,
                ),
            ],
            confidence_level=0.95,
        )
        fig = render_matplotlib_uncertainty(data)
        ax = fig.axes[0]
        ellipses = [p for p in ax.patches if isinstance(p, matplotlib.patches.Ellipse)]
        assert len(ellipses) == 1
        assert ellipses[0].get_linestyle() == "-"
        assert ellipses[0].get_alpha() == pytest.approx(0.15)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Coverage tests for multi_run_handlers.py and callbacks.py
# ---------------------------------------------------------------------------


def _make_uncertainty_spec(
    *,
    group_by: str | list[str] | None = "model",
    label_by: str | None = "concurrency",
) -> PlotSpec:
    """Build a PlotSpec for latency-throughput uncertainty tests."""
    return PlotSpec(
        name="test",
        plot_type=PlotType.LATENCY_THROUGHPUT_UNCERTAINTY,
        metrics=[
            MetricSpec(
                name="request_latency",
                source=DataSource.AGGREGATED,
                axis="x",
                stat="avg",
            ),
            MetricSpec(
                name="output_token_throughput",
                source=DataSource.AGGREGATED,
                axis="y",
                stat="avg",
            ),
        ],
        label_by=label_by,
        group_by=group_by,
    )


class TestLatencyThroughputUncertaintyHandlerCreatePlot:
    """Unit tests for LatencyThroughputUncertaintyHandler.create_plot."""

    def test_create_plot_returns_figure(self) -> None:
        """create_plot with basic DataFrame returns a go.Figure."""
        df = pd.DataFrame(
            {
                "request_latency": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "output_token_throughput": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                "concurrency": [1, 1, 1, 4, 4, 4],
                "model": ["a", "a", "a", "a", "a", "a"],
            }
        )
        spec = _make_uncertainty_spec()
        handler = LatencyThroughputUncertaintyHandler(PlotGenerator())
        fig = handler.create_plot(spec, df, {})
        assert isinstance(fig, go.Figure)

    def test_create_plot_multi_series(self) -> None:
        """create_plot with series_col != point_col produces multiple series."""
        df = pd.DataFrame(
            {
                "request_latency": [1.0, 2.0, 10.0, 20.0],
                "output_token_throughput": [10.0, 20.0, 100.0, 200.0],
                "concurrency": [1, 1, 1, 1],
                "model": ["gpt", "gpt", "llama", "llama"],
            }
        )
        spec = _make_uncertainty_spec(group_by="model")
        handler = LatencyThroughputUncertaintyHandler(PlotGenerator())
        fig = handler.create_plot(spec, df, {})
        assert isinstance(fig, go.Figure)
        # Two models should produce traces for each series
        assert len(fig.data) > 1

    def test_create_plot_list_group_by(self) -> None:
        """create_plot with list-valued group_by on the spec."""
        df = pd.DataFrame(
            {
                "request_latency": [1.0, 2.0, 3.0, 4.0],
                "output_token_throughput": [10.0, 20.0, 30.0, 40.0],
                "concurrency": [1, 1, 2, 2],
                "model": ["a", "a", "a", "a"],
            }
        )
        spec = _make_uncertainty_spec(group_by=["model"])
        handler = LatencyThroughputUncertaintyHandler(PlotGenerator())
        fig = handler.create_plot(spec, df, {})
        assert isinstance(fig, go.Figure)

    def test_can_handle_true_when_columns_exist(self) -> None:
        """can_handle returns True when all metric columns exist."""
        df = pd.DataFrame(
            {
                "request_latency": [1.0],
                "output_token_throughput": [10.0],
            }
        )
        spec = _make_uncertainty_spec()
        handler = LatencyThroughputUncertaintyHandler(PlotGenerator())
        assert handler.can_handle(spec, df) is True

    def test_can_handle_false_when_column_missing(self) -> None:
        """can_handle returns False when a metric column is missing."""
        df = pd.DataFrame({"request_latency": [1.0]})
        spec = _make_uncertainty_spec()
        handler = LatencyThroughputUncertaintyHandler(PlotGenerator())
        assert handler.can_handle(spec, df) is False


class TestBuildUncertaintyPointsAdditional:
    """Additional unit tests for _build_uncertainty_points edge cases."""

    def test_no_group_col_single_group(self) -> None:
        """group_col=None treats all rows as a single group."""
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0],
                "y": [10.0, 20.0, 30.0],
            }
        )
        points = _build_uncertainty_points(
            df, "x", "y", group_col=None, label_col=None, ci_level=0.95
        )
        assert len(points) == 1
        assert points[0].n_runs == 3
        assert points[0].x_mean == pytest.approx(2.0)

    def test_empty_dataframe(self) -> None:
        """Empty DataFrame produces no points."""
        df = pd.DataFrame({"x": pd.Series(dtype=float), "y": pd.Series(dtype=float)})
        points = _build_uncertainty_points(
            df, "x", "y", group_col=None, label_col=None, ci_level=0.95
        )
        assert len(points) == 0


class TestBuildUncertaintyFigureAdditional:
    """Additional unit tests for _build_uncertainty_figure in callbacks.py."""

    def test_concurrency_column_preferred_for_grouping(self) -> None:
        """When concurrency column is present, it is used for point-level grouping."""
        df = pd.DataFrame(
            {
                "x_metric": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
                "y_metric": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
                "concurrency": [1, 1, 1, 4, 4, 4],
                "other_group": ["a", "a", "a", "b", "b", "b"],
            }
        )
        pg = PlotGenerator()
        fig = _build_uncertainty_figure(
            df,
            "x_metric",
            "y_metric",
            pg,
            actual_group_by="other_group",
            actual_label_by=None,
            plot_config_dict={},
            title="Test",
            x_label="X",
            y_label="Y",
        )
        assert isinstance(fig, go.Figure)
        # 2 series (other_group a and b), each with 1 concurrency point
        mean_traces = [
            t for t in fig.data if t.error_x is not None and t.error_x.array is not None
        ]
        assert len(mean_traces) == 2

    def test_no_concurrency_falls_back_to_actual_group_by(self) -> None:
        """Without concurrency column, actual_group_by becomes the series key."""
        df = pd.DataFrame(
            {
                "x_metric": [1.0, 2.0, 10.0, 20.0],
                "y_metric": [10.0, 20.0, 100.0, 200.0],
                "model": ["a", "a", "b", "b"],
            }
        )
        pg = PlotGenerator()
        fig = _build_uncertainty_figure(
            df,
            "x_metric",
            "y_metric",
            pg,
            actual_group_by="model",
            actual_label_by=None,
            plot_config_dict={},
            title="Test",
            x_label="X",
            y_label="Y",
        )
        assert isinstance(fig, go.Figure)
        # No concurrency → model becomes series key → 2 series, 1 point each
        mean_traces = [
            t for t in fig.data if t.error_x is not None and t.error_x.array is not None
        ]
        assert len(mean_traces) == 2

    def test_valid_ci_level_090_shows_in_legend(self) -> None:
        """ci_level=0.90 produces a legend entry with '90%'."""
        df = pd.DataFrame(
            {
                "x_metric": [1.0, 2.0, 3.0],
                "y_metric": [10.0, 20.0, 30.0],
                "concurrency": [1, 1, 1],
            }
        )
        pg = PlotGenerator()
        fig = _build_uncertainty_figure(
            df,
            "x_metric",
            "y_metric",
            pg,
            actual_group_by=None,
            actual_label_by=None,
            plot_config_dict={"ci_level": 0.90},
            title="Test",
            x_label="X",
            y_label="Y",
        )
        assert isinstance(fig, go.Figure)
        legend_names = [t.name for t in fig.data if t.name and "Confidence" in t.name]
        assert any("90%" in name for name in legend_names), (
            f"Expected 90% in legend, got: {legend_names}"
        )
