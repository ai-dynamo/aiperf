# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Custom Resource coordinates for CustomObjectsApi calls.

Canonical source for (group, version, plural) triples used across AIPerf.
Every call site that reads, writes, or refers to a CRD should import from
this module -- no duplicate constants elsewhere.
"""

# AIPerfJob (the AIPerf-owned CR)
AIPERF_JOB_GROUP = "aiperf.nvidia.com"
AIPERF_JOB_VERSION = "v1alpha1"
AIPERF_JOB_PLURAL = "aiperfjobs"
AIPERF_JOB_API_VERSION = f"{AIPERF_JOB_GROUP}/{AIPERF_JOB_VERSION}"

# Backwards-compat aliases -- previous names in kubernetes/constants.py
# and cli_commands/kube/*.py that other modules already import.
AIPERF_GROUP = AIPERF_JOB_GROUP
AIPERF_VERSION = AIPERF_JOB_VERSION
AIPERF_PLURAL = AIPERF_JOB_PLURAL
AIPERF_API_VERSION = AIPERF_JOB_API_VERSION

# JobSet (external -- jobset-operator)
JOBSET_GROUP = "jobset.x-k8s.io"
JOBSET_VERSION = "v1alpha2"
JOBSET_PLURAL = "jobsets"
JOBSET_API_VERSION = f"{JOBSET_GROUP}/{JOBSET_VERSION}"

# Kueue (external -- optional queue operator used by operator preflight)
KUEUE_GROUP = "kueue.x-k8s.io"
KUEUE_VERSION = "v1beta1"
KUEUE_LOCALQUEUE_PLURAL = "localqueues"
