# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operator chaos-testing suite.

Tests under this package inject faults against a running AIPerfJob
and verify that the operator reaches the documented end state.
Every test assumes the standard ``tests/kubernetes/conftest.py``
fixtures (``local_cluster``, ``operator_ready``, ``kubectl``) are
available. Marked ``k8s_slow`` because scenarios need generous
wait windows.
"""
