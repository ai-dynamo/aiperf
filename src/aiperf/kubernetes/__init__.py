# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kubernetes deployment module for AIPerf.

This module provides tools for deploying AIPerf as a distributed benchmark
across multiple Kubernetes pods using JobSet. Configuration is available via
environment variables with the AIPERF_K8S_ prefix through K8sEnvironment.
"""
