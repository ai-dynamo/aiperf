# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sweep-controller pod package.

Runs inside the JobSet created by the operator's @kopf.on.create handler
for an AIPerfSweep CR. Reads the CR, builds a BenchmarkPlan, and uses
MultiRunOrchestrator + K8sChildJobExecutor to iterate variations x trials,
creating child AIPerfJob CRs deterministically.
"""
