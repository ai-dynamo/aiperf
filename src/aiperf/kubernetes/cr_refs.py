# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Custom Resource coordinates for CustomObjectsApi calls.

These constants replace the kr8s ``new_class(...)`` wrappers. Every
``CustomObjectsApi.*_namespaced_custom_object`` call in the codebase
takes the matching (group, version, plural) triple from this module.
"""

# AIPerfJob (the AIPerf-owned CR)
AIPERF_JOB_GROUP = "aiperf.nvidia.com"
AIPERF_JOB_VERSION = "v1alpha1"
AIPERF_JOB_PLURAL = "aiperfjobs"

# JobSet (external — jobset-operator)
JOBSET_GROUP = "jobset.x-k8s.io"
JOBSET_VERSION = "v1alpha2"
JOBSET_PLURAL = "jobsets"
