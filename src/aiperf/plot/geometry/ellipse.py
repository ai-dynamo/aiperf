# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure ellipse polygon geometry — no rendering dependencies."""

import logging
import math

import numpy as np
from scipy.stats import chi2

logger = logging.getLogger(__name__)

_EIGENVALUE_FLOOR = 1e-12


def compute_ellipse_vertices(
    cov: np.ndarray,
    center: tuple[float, float],
    confidence_level: float,
    n_vertices: int = 64,
) -> list[tuple[float, float]]:
    """Compute closed polygon vertices for a confidence ellipse.

    Uses eigendecomposition of the 2x2 covariance matrix to determine
    semi-axes and rotation, scaled by chi-squared quantile.

    Args:
        cov: 2x2 covariance matrix (must be positive semi-definite).
        center: (x, y) center of the ellipse.
        confidence_level: Confidence level (e.g., 0.95).
        n_vertices: Number of polygon vertices (default 64).

    Returns:
        List of (x, y) tuples forming a closed polygon (first == last).
    """
    if n_vertices < 3:
        raise ValueError(f"n_vertices must be >= 3, got {n_vertices}")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    if np.any(eigenvalues < 0):
        logger.warning(
            "Non-positive-definite covariance matrix detected; "
            "clamping negative eigenvalues to %e",
            _EIGENVALUE_FLOOR,
        )
        eigenvalues = np.maximum(eigenvalues, _EIGENVALUE_FLOOR)

    scale = math.sqrt(chi2.ppf(confidence_level, df=2))
    a = scale * math.sqrt(float(eigenvalues[1]))
    b = scale * math.sqrt(float(eigenvalues[0]))
    theta = math.atan2(float(eigenvectors[1, 1]), float(eigenvectors[0, 1]))

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    cx, cy = center

    vertices: list[tuple[float, float]] = []
    for i in range(n_vertices):
        t = 2.0 * math.pi * i / n_vertices
        x_raw = a * math.cos(t)
        y_raw = b * math.sin(t)
        x = cx + x_raw * cos_t - y_raw * sin_t
        y = cy + x_raw * sin_t + y_raw * cos_t
        vertices.append((x, y))

    vertices.append(vertices[0])
    return vertices


def compute_axis_aligned_ellipse_vertices(
    center: tuple[float, float],
    x_radius: float,
    y_radius: float,
    n_vertices: int = 64,
) -> list[tuple[float, float]]:
    """Compute vertices for an axis-aligned ellipse from CI bounds.

    Args:
        center: (x, y) center.
        x_radius: Semi-axis extent along x.
        y_radius: Semi-axis extent along y.
        n_vertices: Number of polygon vertices.

    Returns:
        List of (x, y) tuples forming a closed polygon (first == last).
    """
    if n_vertices < 3:
        raise ValueError(f"n_vertices must be >= 3, got {n_vertices}")

    cx, cy = center
    vertices: list[tuple[float, float]] = []
    for i in range(n_vertices):
        t = 2.0 * math.pi * i / n_vertices
        vertices.append((cx + x_radius * math.cos(t), cy + y_radius * math.sin(t)))
    vertices.append(vertices[0])
    return vertices
