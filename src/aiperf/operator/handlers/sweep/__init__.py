# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kopf handlers for AIPerfSweep CRs.

Three responsibilities, mirrored as submodules:
- create: validate spec, provision RBAC, create sweep-controller JobSet.
- lifecycle: mirror spec.cancel into status.conditions.
- child_rollup: bubble AIPerfJob.status.phase transitions into the parent.
"""
