# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU integration tests for Kubernetes E2E testing.

DO NOT add __init__.py to subdirectories (vllm/, dynamo/).  This package is
the scope boundary for ``scope="package"`` fixtures defined in gpu/conftest.py
(gpu_cluster, kubectl, gpu_cluster_base, etc.).  Adding __init__.py to a
subdirectory turns it into a separate package, causing those fixtures to be
instantiated per-subdirectory (e.g. cluster created twice).

To avoid test module name collisions without __init__.py, prefix test files
with the subdirectory name: ``test_vllm_benchmark.py``, not ``test_benchmark.py``.
"""
